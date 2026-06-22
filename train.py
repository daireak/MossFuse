# -*- coding: utf-8 -*-

import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
from MossFuseNet import MossFuse
import datetime
import itertools
import sys
import time
import cv2
import hdf5storage as hdf5
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import cc, PSNR_SSIM_cal, cal_decomp_loss, save_final_results

'''
------------------------------------------------------------------------------
Argument Parser Configuration
------------------------------------------------------------------------------
'''
parser = argparse.ArgumentParser(description='Train MossFuse for HMIF')
# Dataset settings
parser.add_argument('--dataset_type', type=str, default='single', choices=['multi', 'single'], help='multi (e.g., CAVE) or single (e.g., PU)')
parser.add_argument('--data_path', type=str, required=True, help='Directory for multi-image, or .mat file path for single-image')
parser.add_argument('--r_path', type=str, required=True, help='Path to the spectral response function .mat file')
parser.add_argument('--mat_key', type=str, default='paviaU', help='Key of the HSI data in .mat file (e.g. rad, paviaU)')
parser.add_argument('--r_key', type=str, default='R', help='Key of the SRF data in .mat file (e.g. resp, R)')

# Network settings
parser.add_argument('--scale', type=int, default=4, help='Spatial downsampling scale factor')
parser.add_argument('--channels_hsi', type=int, default=103, help='Number of HSI bands')
parser.add_argument('--channels_msi', type=int, default=4, help='Number of MSI bands')
parser.add_argument('--patch_size', type=int, default=64, help='Training patch size')

# Training settings
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs') # 默认改为300
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--exp_name', type=str, default='PU_Exp', help='Experiment name for saving logs and models')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

args = parser.parse_args()

'''
------------------------------------------------------------------------------
Configure our network & environment
------------------------------------------------------------------------------
'''
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 动态加载对应的数据集类
if args.dataset_type == 'multi':
    from dataset import MultiImageDatasetTrain, MultiImageDatasetTest
    all_files = [f for f in os.listdir(args.data_path) if f.endswith('.mat')]
    all_files.sort()
    split_idx = int(0.8 * len(all_files))
    train_files, test_files = all_files[:split_idx], all_files[split_idx:]
    
    Hyper_train = MultiImageDatasetTrain(args.data_path, args.r_path, train_files, mat_key=args.mat_key, r_key=args.r_key, scale=args.scale, patch_size=args.patch_size)
    Hyper_test = MultiImageDatasetTest(args.data_path, args.r_path, test_files, mat_key=args.mat_key, r_key=args.r_key, scale=args.scale)
else:
    from dataset import SingleImageDatasetTrain, SingleImageDatasetTest
    Hyper_train = SingleImageDatasetTrain(args.data_path, args.r_path, mat_key=args.mat_key, r_key=args.r_key, scale=args.scale, patch_size=args.patch_size)
    Hyper_test = SingleImageDatasetTest(args.data_path, args.r_path, mat_key=args.mat_key, r_key=args.r_key, scale=args.scale)

trainloader = DataLoader(Hyper_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
testloader = DataLoader(Hyper_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)

# 动态初始化模型
model = nn.DataParallel(MossFuse(dim=48, num_blocks=3, scale=args.scale, channels_MSI=args.channels_msi, channels_HSI=args.channels_hsi)).to(device)

optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)

# 【重要修复 1】: T_max 必须等于总 Epoch 数，否则学习率几乎不下降
T_max = args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-6, last_epoch=-1) 

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
log_dir_name = f"models/Tensorboard_{args.exp_name}_{timestamp}_scale{args.scale}_lr{args.lr}"
writer = SummaryWriter(log_dir=log_dir_name)
writer.add_text('Experiment Config: ', str(args))

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()

if not os.path.exists("models"):
    os.makedirs("models")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True

# 全局动态加载 SRF 用于 Loss 计算
try:
    res = scio.loadmat(args.r_path)[args.r_key]
except:
    res = hdf5.loadmat(args.r_path)[args.r_key]

if res.shape[0] > res.shape[1]:
    res = np.transpose(res, (1, 0))

if res.shape[1] > args.channels_hsi:
    diff = res.shape[1] - args.channels_hsi
    print(f"⚠️ [Loss计算] 全局R矩阵波段({res.shape[1]}) > HSI设定波段({args.channels_hsi})，自动舍弃【末尾】 {diff} 个波段！")
    res = res[:, :-diff] 

row_sum = np.sum(res, axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
res = res / row_sum
srf_g = torch.Tensor(res).to(device)

gaussian_kernel = cv2.getGaussianKernel(7, 2)
psf_g = torch.Tensor(gaussian_kernel * gaussian_kernel.T).to(device)


for epoch in range(args.epochs):
    ''' train '''
    cc_sim_ = []; cc_unsim_ = []; mse_MSI_ = []; mse_HSI_ = []
    mse_HSI_R_ = []; mse_MSI_R_ = []; mse_srf_ = []; mse_psf_ = []
    mse_lr_msi_ = []; mse_HSI_R_fHSI_ = []; mse_MSI_R_fHSI_ = []
    mse_HR_HSI_ = []; loss_decomp_ = []

    for msi, hsi, hsi_g in tqdm(trainloader):
        hsi_ = torch.nn.functional.interpolate(hsi, scale_factor=(args.scale, args.scale), mode='bilinear')
        msi, hsi, hsi_, hsi_g = msi.to(device), hsi.to(device), hsi_.to(device), hsi_g.to(device)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral, msi_out, hsi_out, lr_msi_fhsi, lr_msi_fmsi, lr_msi_out, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(msi, hsi)

        cc_loss_sim = L1Loss(msi_spatial_spectral, hsi_spatial_spectral)
        cc_loss_unsim = cc(msi_spatial, hsi_spectral)
        
        mse_loss_msi = L1Loss(msi, msi_out)
        
        # 【重要修复 2】: 解决下采样网格亚像素级不对齐的问题，统一改为 0::scale
        mse_loss_hsi = L1Loss(hsi, hsi_out[:, :, 0::args.scale, 0::args.scale])

        mse_LR_MSI = L1Loss(lr_msi_fhsi, lr_msi_fmsi)
        lr_msi_fhsi = torch.nn.functional.interpolate(lr_msi_fhsi, scale_factor=(args.scale, args.scale), mode='bilinear')
        lr_msi_fmsi = torch.nn.functional.interpolate(lr_msi_fmsi, scale_factor=(args.scale, args.scale), mode='bilinear')
        mse_HSI_R = L1Loss(lr_msi_fhsi, lr_msi_out)
        mse_MSI_R = L1Loss(lr_msi_fmsi, lr_msi_out)

        loss_decomp = cal_decomp_loss(msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral)

        mse_srf = L1Loss(srf_g, srf)
        mse_psf = L1Loss(psf_g, psf)

        mse_hsi_fhrHSI = L1Loss(hsi, hsi_fhsi)
        mse_msi_fhrHSI = L1Loss(msi, msi_fhsi)
        mse_HSI = L1Loss(hsi_g, HR_HSI)

        cc_sim_.append(cc_loss_sim.data.cpu().numpy())
        cc_unsim_.append(cc_loss_unsim.data.cpu().numpy())
        mse_MSI_.append(mse_loss_msi.data.cpu().numpy())
        mse_HSI_.append(mse_loss_hsi.data.cpu().numpy())
        mse_HSI_R_.append(mse_HSI_R.data.cpu().numpy())
        mse_MSI_R_.append(mse_MSI_R.data.cpu().numpy())
        mse_srf_.append(mse_srf.data.cpu().numpy())
        mse_psf_.append(mse_psf.data.cpu().numpy())
        mse_lr_msi_.append(mse_LR_MSI.data.cpu().numpy())
        mse_HSI_R_fHSI_.append(mse_hsi_fhrHSI.data.cpu().numpy())
        mse_MSI_R_fHSI_.append(mse_msi_fhrHSI.data.cpu().numpy())
        mse_HR_HSI_.append(mse_HSI.data.cpu().numpy())
        loss_decomp_.append(loss_decomp.data.cpu().numpy())

        # 【重要修复 3】: 将遗漏的自监督约束损失（L_SCT）重新加回总 Loss 中
        loss_SCT = mse_loss_msi + mse_loss_hsi + mse_HSI_R + mse_MSI_R
        loss1 = 0.1*loss_decomp + mse_hsi_fhrHSI + mse_msi_fhrHSI + loss_SCT
        
        for name, param in model.named_parameters():
            if ("blind" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
        loss1.backward(retain_graph=True)

        loss2 = mse_LR_MSI
        for name, param in model.named_parameters():
            if ("blind" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss2.backward()

        for name, param in model.named_parameters():
            param.requires_grad = True

        optimizer.step()
        
    print("Train epoch:%d, sim:%.5f, decom:%.5f, MSI:%.5f, HSI:%.5f, HSI_R:%.5f, MSI_R:%.5f, srf:%.5f, psf:%.5f, lr_msi:%.5f, HR-MSI_R:%.5f, LR-HSI_R:%.5f, HSI:%.5f"%
          (epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(loss_decomp_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)), 
           np.mean(np.array(mse_lr_msi_)), np.mean(np.array(mse_MSI_R_fHSI_)), np.mean(np.array(mse_HSI_R_fHSI_)), np.mean(np.array(mse_HR_HSI_))))

    writer.add_scalar('train/sim', np.mean(np.array(cc_sim_)), epoch)
    writer.add_scalar('train/loss_decomp_', np.mean(np.array(loss_decomp_)), epoch)
    writer.add_scalar('train/MSI_Decoder', np.mean(np.array(mse_MSI_)), epoch)
    writer.add_scalar('train/HSI_Decoder', np.mean(np.array(mse_HSI_)), epoch)
    writer.add_scalar('train/SRF', np.mean(np.array(mse_srf_)), epoch)
    writer.add_scalar('train/PSF', np.mean(np.array(mse_psf_)), epoch)

    # Test Phase
    loss = 0
    cc_sim_ = []; cc_unsim_ = []; mse_MSI_ = []; mse_HSI_ = []
    mse_HSI_R_ = []; mse_MSI_R_ = []; mse_srf_ = []; mse_psf_ = []
    mse_lr_msi_ = []; mse_HSI_R_fHSI_ = []; mse_MSI_R_fHSI_ = []
    mse_HR_HSI_ = []; PSNR_HR_HSI_ = []; SSIM_HR_HSI_ = []; loss_decomp_ = []

    for msi, hsi, hsi_g, *img_name in tqdm(testloader):
        with torch.no_grad():
            msi, hsi, hsi_, hsi_g = msi.cuda(), hsi.cuda(), hsi_.cuda(), hsi_g.cuda()
            hsi_ = torch.nn.functional.interpolate(hsi, scale_factor=(args.scale, args.scale), mode='bilinear')
            model.eval()
            msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral, msi_out, hsi_out, lr_msi_fhsi, lr_msi_fmsi, lr_msi_out, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(msi, hsi)

            cc_loss_sim = cc(msi_spatial_spectral, hsi_spatial_spectral)
            cc_loss_unsim = cc(msi_spatial, hsi_spectral)
            
            mse_loss_msi = L1Loss(msi, msi_out)
            # 【重要修复 2 延续】: 测试集也要统一为 0::scale
            mse_loss_hsi = L1Loss(hsi, hsi_out[:, :, 0::args.scale, 0::args.scale])

            mse_LR_MSI = L1Loss(lr_msi_fhsi, lr_msi_fmsi)
            lr_msi_fhsi = torch.nn.functional.interpolate(lr_msi_fhsi, scale_factor=(args.scale, args.scale), mode='bilinear')
            lr_msi_fmsi = torch.nn.functional.interpolate(lr_msi_fmsi, scale_factor=(args.scale, args.scale), mode='bilinear')
            mse_HSI_R = L1Loss(lr_msi_fhsi, lr_msi_out)
            mse_MSI_R = L1Loss(lr_msi_fmsi, lr_msi_out)

            loss_decomp = cal_decomp_loss(msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral)
            mse_srf = L1Loss(srf_g, srf)
            mse_psf = L1Loss(psf_g, psf)            

            mse_hsi_fhrHSI = L1Loss(hsi, hsi_fhsi)
            mse_msi_fhrHSI = L1Loss(msi, msi_fhsi)
            mse_HSI = L1Loss(hsi_g, HR_HSI)
            psnr_HSI, ssim_HSI = PSNR_SSIM_cal(hsi_g, HR_HSI)
            
            if epoch == args.epochs - 1:
                pred_np = HR_HSI.squeeze().detach().cpu().numpy()
                gt_np = hsi_g.squeeze().detach().cpu().numpy()
                if pred_np.ndim == 3:
                    pred_np = pred_np.transpose(1, 2, 0)
                    gt_np = gt_np.transpose(1, 2, 0)
                elif pred_np.ndim == 4:
                    pred_np = pred_np[0].transpose(1, 2, 0)
                    gt_np = gt_np[0].transpose(1, 2, 0)
                save_final_results(gt_np, pred_np, epoch, sf=args.scale)
            
            cc_sim_.append(np.array(cc_loss_sim.data.cpu()))
            cc_unsim_.append(np.array(cc_loss_unsim.data.cpu()))
            mse_MSI_.append(np.array(mse_loss_msi.data.cpu()))
            mse_HSI_.append(np.array(mse_loss_hsi.data.cpu()))
            mse_HSI_R_.append(np.array(mse_HSI_R.data.cpu()))
            mse_MSI_R_.append(np.array(mse_MSI_R.data.cpu()))
            mse_srf_.append(np.array(mse_srf.data.cpu()))
            mse_psf_.append(np.array(mse_psf.data.cpu()))
            mse_lr_msi_.append(mse_LR_MSI.data.cpu().numpy())
            mse_HSI_R_fHSI_.append(mse_hsi_fhrHSI.data.cpu().numpy())
            mse_MSI_R_fHSI_.append(mse_msi_fhrHSI.data.cpu().numpy())
            mse_HR_HSI_.append(mse_HSI.data.cpu().numpy())
            PSNR_HR_HSI_.append(psnr_HSI)
            SSIM_HR_HSI_.append(ssim_HSI)
            loss_decomp_.append(loss_decomp.data.cpu().numpy())

    print("Test  epoch:%d, sim:%.5f, decom:%.5f, MSI:%.5f, HSI:%.5f, HSI_R:%.5f, MSI_R:%.5f, srf:%.5f, psf:%.5f, lr_msi:%.5f, HR-MSI_R:%.5f, LR-HSI_R:%.5f, HSI:%.5f, psnr:%.5f, ssim:%.5f"%
          (epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(loss_decomp_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)), 
           np.mean(np.array(mse_lr_msi_)), np.mean(np.array(mse_MSI_R_fHSI_)), np.mean(np.array(mse_HSI_R_fHSI_)), 
           np.mean(np.array(mse_HR_HSI_)), np.mean(np.array(PSNR_HR_HSI_)), np.mean(np.array(SSIM_HR_HSI_))))

    writer.add_scalar('test/PSNR', np.mean(np.array(PSNR_HR_HSI_)), epoch)
    writer.add_scalar('test/SSIM', np.mean(np.array(SSIM_HR_HSI_)), epoch)

    scheduler.step()  
    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
    
    checkpoint = {
        'Model_stage1': model.state_dict(),
    }
    pth_dir = os.path.join("models", "pth", f"{args.exp_name}_{timestamp}")
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)
    torch.save(checkpoint, os.path.join(pth_dir, f"{args.exp_name}_epoch_{epoch}.pth"))