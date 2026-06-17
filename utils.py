import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.nn.functional import cosine_similarity
from sewar.full_ref import uqi
def cc(img_fake, img_real):
    """
    基于 PyTorch 实现的 Cross Correlation (皮尔逊相关系数)
    输入形状要求: (B, C, H, W) 张量
    """
    # 确保输入是 PyTorch 张量
    B, C, H, W = img_fake.shape
    
    # 将空间维度 (H, W) 展平，形状变为 (B, C, H*W)
    v1 = img_fake.view(B, C, -1)
    v2 = img_real.view(B, C, -1)
    
    # 沿着空间维度求均值，形状变为 (B, C, 1)
    v1_mean = v1.mean(dim=2, keepdim=True)
    v2_mean = v2.mean(dim=2, keepdim=True)
    
    # 减去均值
    v1 = v1 - v1_mean
    v2 = v2 - v2_mean
    
    # 分子：按元素相乘后在空间维度求和
    num = (v1 * v2).sum(dim=2)
    
    # 分母：各自平方和的乘积，再开方
    den = torch.sqrt((v1 ** 2).sum(dim=2) * (v2 ** 2).sum(dim=2))
    
    # 避免除以 0 (加上一个极小值 1e-8)
    den = torch.clamp(den, min=1e-8)
    
    # 计算每个通道的 CC 值，然后对 Batch 和 Channels 求平均
    cc_val = num / den
    
    return cc_val.mean()

def record_loss(loss_csv,epoch, cc_B, cc_D, mse_MSI, mse_HSI, mse_HSI_R, mse_MSI_R, mse_srf, mse_psf):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, cc_B, cc_D, mse_MSI, mse_HSI, mse_HSI_R, mse_MSI_R, mse_srf, mse_psf))
    loss_csv.flush()    
    loss_csv.close
    
import os

def show(epoch, srf, srf_g, psf, psf_g):
    # 使用 detach() 顺便解决 NumPy 2.0 的警告问题
    srf = srf.detach().cpu().numpy()
    srf_g = srf_g.detach().cpu().numpy()
    psf = psf.detach().cpu().numpy()
    psf_g = psf_g.detach().cpu().numpy()
    
    os.makedirs('./results', exist_ok=True) # 确保 results 文件夹存在
    
    # show SRF
    channel = range(srf.shape[1]) # 动态适配波段数
    plt.figure(figsize=(10, 6), facecolor='lightgray', edgecolor='black')
    plt.plot(channel, srf[0,:], marker='o', linestyle='--', color='b')
    plt.plot(channel, srf_g[0,:], marker='o', linestyle='-', color='b')
    plt.plot(channel, srf[1,:], marker='o', linestyle='--', color='g')
    plt.plot(channel, srf_g[1,:], marker='o', linestyle='-', color='g')
    plt.plot(channel, srf[2,:], marker='o', linestyle='--', color='r')
    plt.plot(channel, srf_g[2,:], marker='o', linestyle='-', color='r')
    plt.title('Spectral Response Curve')
    plt.xlabel('Spectral')
    plt.ylabel('Response')
    plt.grid(True)
    plt.savefig(f'./results/src_epoch_{epoch}.png') # 加上 epoch 防止覆盖
    
    # show PSF
    plt.figure(figsize=(10, 4), facecolor='lightgray', edgecolor='black')
    plt.subplot(131), plt.imshow(psf, cmap='hot', interpolation='nearest'), plt.title('PSF')
    plt.subplot(132), plt.imshow(psf_g, cmap='hot', interpolation='nearest'), plt.title('PSF_GT')
    plt.subplot(133), plt.imshow(np.abs(psf-psf_g)*10**4, cmap='hot', interpolation='nearest', vmin=0, vmax=1), plt.title('PSF_error x 10**4')
    plt.savefig(f'./results/psf_epoch_{epoch}.png') # 加上 epoch 防止覆盖
    plt.close('all')
def load_model(model, model_name, model_var='Model_stage1'):
    model_param = torch.load(model_name, weights_only=True)[model_var]
    model_dict = {}
    for k1, k2 in zip(model.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model.load_state_dict(model_dict)
    return model.cuda()

def PSNR_SSIM_cal(gt, rec):
    gt = gt.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    psnr = cal_psnr(gt[0,:,:,:], rec[0,:,:,:])
    gt = np.transpose(gt,(0,2,3,1))[0,:,:,:]
    rec = np.transpose(rec,(0,2,3,1))[0,:,:,:]
    ssim = compare_ssim(gt, rec, K1 = 0.01, K2 = 0.03, channel_axis=-1, data_range=1)
    return psnr, np.mean(np.array(ssim))

def cal_cos_loss(l1, l2):
    l1 = l1.view(-1)
    l2 = l2.view(-1)
    similarity = cosine_similarity(l1, l2, dim=-1)
    return similarity


def cal_decomp_loss(RGB_spatial_spectral, RGB_spatial, HSI_spatial_spectral, HSI_spectral):
    positive = torch.exp(cal_cos_loss(RGB_spatial_spectral, HSI_spatial_spectral))
    negative = torch.exp(cal_cos_loss(RGB_spatial_spectral, RGB_spatial)) + torch.exp(cal_cos_loss(HSI_spatial_spectral, HSI_spectral)) + torch.exp(cal_cos_loss(RGB_spatial, HSI_spectral))
    decomp_loss = -torch.log(positive/(positive+negative))
    return decomp_loss


def cal_psnr(label, output):

    img_c, img_w, img_h = label.shape
    ref = label.reshape(img_c, -1)
    tar = output.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(1 / msr)
    out_mean = np.mean(psnrall)
    # return out_mean, max1
    return out_mean


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_valid(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,6])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count

class Loss_valid(nn.Module):
    def __init__(self, scale=8):
        super(Loss_valid, self).__init__()
        self.scale=scale

    def forward(self, label_image, rec_image):
        self.batch_size = label_image.shape[0]
        assert self.batch_size == 1
        self.label = label_image.data.cpu().squeeze(0).numpy()
        self.output = rec_image.data.cpu().squeeze(0).numpy()
        self.output = np.clip(self.output, 0, 1)
        valid_error = np.zeros([1, 6])

        valid_error[0, 0] = self.ssim()
        valid_error[0, 1] = self.cal_rmse()
        valid_error[0, 2] = self.cal_psnr()
        valid_error[0, 3] = self.cal_ergas()
        valid_error[0, 4] = self.sam()
        valid_error[0, 5] = self.cal_uqi()
        return valid_error

    def cal_mrae(self):
        error = np.abs(self.output - self.label) / self.label
        # error = torch.abs(outputs - label)
        mrae = np.mean(error.reshape(-1))
        return mrae

    def cal_rmse(self):
        rmse = np.sqrt(np.mean((self.label-self.output)**2))
        return rmse

    def cal_psnr(self):
        
        assert self.label.ndim == 3 and self.output.ndim == 3

        img_c, img_w, img_h = self.label.shape
        ref = self.label.reshape(img_c, -1)
        tar = self.output.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max1 = np.max(ref, 1)

        psnrall = 10 * np.log10(1 / msr)
        out_mean = np.mean(psnrall)
        # return out_mean, max1
        return out_mean

    def cal_ergas(self, scale=32):
        d = self.label - self.output
        ergasroot = 0
        for i in range(d.shape[0]):
            ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
        ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
        return ergas

    def cal_sam(self):
        assert self.label.ndim == 3 and self.label.shape == self.label.shape

        c, w, h = self.label.shape
        x_true = self.label.reshape(c, -1)
        x_pred = self.output.reshape(c, -1)

        x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

        sam = np.arccos(sam) * 180 / np.pi
        # sam = np.arccos(sam)
        mSAM = sam.mean()
        var_sam = np.var(sam)
        # return mSAM, var_sam
        return mSAM

    def cal_ssim(self, data_range=1, multidimension=False):
        """
        :param x_true:
        :param x_pred:
        :param data_range:
        :param multidimension:
        :return:
        """
        mssim = [
            compare_ssim(X=self.label[i, :, :], Y=self.output[i, :, :], data_range=data_range, multidimension=multidimension)
            for i in range(self.label.shape[0])]
        return np.mean(mssim)

    def cal_uqi(self):
        fout = np.transpose(self.output, [1,2,0])
        hsi_g = np.transpose(self.label, [1,2,0])
        uqi_ = uqi(hsi_g, fout)
        return uqi_

    def ssim(self):
        fout_0 = np.transpose(self.output, [1,2,0])
        hsi_g_0 = np.transpose(self.label, [1,2,0])
        # ssim_result = compare_ssim(fout_0, hsi_g_0, data_range=1)
        ssim = compare_ssim(hsi_g_0, fout_0, K1 = 0.01, K2 = 0.03, channel_axis=-1, data_range=1)
        return ssim
    
    def psnr(self):
        fout = self.output
        hsi_g = self.label
        psnr_g = []
        for i in range(31):
            psnr_g.append(compare_psnr(hsi_g[i,:,:],fout[i,:,:]))
        return np.mean(np.array(psnr_g))

    def sam(self):
        """
        cal SAM between two images
        :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
        :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
        :return: Spectral Angle Mapper between `recovered` and `groundTruth`.
        """
        groundTruth = np.transpose(self.label, [1,2,0])
        recovered = np.transpose(self.output, [1,2,0])
        recovered = np.clip(recovered, 0.00001, 1)
        assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

        nom = np.sum(groundTruth * recovered, 2)
        denom1 = np.sqrt(np.sum(groundTruth**2, 2))
        denom2 = np.sqrt(np.sum(recovered ** 2, 2))
        sam = np.arccos(np.divide(nom, denom1*denom2))
        sam = np.divide(sam, np.pi) * 180.0
        sam = np.mean(sam)

        return sam

import os
import matplotlib.pyplot as plt
import scipy.signal
from skimage.metrics import structural_similarity as ssim

# ==================== 用户自定义的 6 项指标与可视化 ====================
def bandwise_psnr(img_real, img_fake, data_range=1.0):
    mse_bands = np.mean((img_real - img_fake) ** 2, axis=(0, 1))
    psnr_bands = np.zeros_like(mse_bands)
    zero_mask = (mse_bands == 0)
    non_zero_mask = ~zero_mask
    psnr_bands[non_zero_mask] = 10 * np.log10((data_range ** 2) / mse_bands[non_zero_mask])
    psnr_bands[zero_mask] = 100.0
    return np.mean(psnr_bands)

def ergas(img_fake, img_real, scale_factor):
    img_fake, img_real = np.clip(img_fake, 0.0, 1.0), np.clip(img_real, 0.0, 1.0)
    channels = img_real.shape[2]
    inner_sum = sum(((np.sqrt(np.mean((img_real[:, :, i] - img_fake[:, :, i]) ** 2)) / (np.mean(img_real[:, :, i]) + 1e-8)) ** 2) for i in range(channels) if np.mean(img_real[:, :, i]) != 0)
    return 100 / scale_factor * np.sqrt(inner_sum / channels)

def cross_correlation_metric(img_fake, img_real):
    channels = img_real.shape[2]
    cc_val = 0
    for i in range(channels):
        v1, v2 = img_fake[:, :, i].flatten(), img_real[:, :, i].flatten()
        v1, v2 = v1 - np.mean(v1), v2 - np.mean(v2)
        den = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
        if den != 0: cc_val += np.sum(v1 * v2) / den
    return cc_val / channels

def sam_metric(img1, img2):
    img1, img2 = img1.reshape(-1, img1.shape[-1]), img2.reshape(-1, img2.shape[-1])
    cos_theta = np.clip(np.sum(img1 * img2, axis=-1) / (np.linalg.norm(img1, axis=-1) * np.linalg.norm(img2, axis=-1) + 1e-8), -1, 1)
    return np.mean(np.arccos(cos_theta)) * 180 / np.pi

def quality_assessment(S_true, Z_pred, sf):
    Z_pred, S_true = np.clip(Z_pred, 0.0, 1.0), np.clip(S_true, 0.0, 1.0)
    return (bandwise_psnr(S_true, Z_pred, data_range=1.0), 
            ssim(S_true, Z_pred, channel_axis=-1, data_range=1.0), 
            sam_metric(S_true, Z_pred), ergas(Z_pred, S_true, sf), 
            cross_correlation_metric(Z_pred, S_true), np.mean(np.abs(Z_pred - S_true) * 255.0))

def matlab_style_rgb(img_3d, bands):
    rgb = img_3d[:, :, bands].copy().astype(np.float32)
    for i in range(3):
        c_min, c_max = rgb[:, :, i].min(), rgb[:, :, i].max()
        rgb[:, :, i] = (rgb[:, :, i] - c_min) / (c_max - c_min + 1e-8)
    return rgb

def save_final_results(S_true, Z_pred, epoch, sf=4, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    # 计算 6 个指标
    psnr_v, ssim_v, sam_v, ergas_v, cc_v, dd_v = quality_assessment(S_true, Z_pred, sf)
    
    print(f"\n{'='*20} 最终评估结果 (Epoch: {epoch}) {'='*20}")
    print(f"{'PSNR':<10} | {psnr_v:<10.4f}")
    print(f"{'SSIM':<10} | {ssim_v:<10.4f}")
    print(f"{'SAM':<10} | {sam_v:<10.4f}")
    print(f"{'ERGAS':<10} | {ergas_v:<10.4f}")
    print(f"{'CC':<10} | {cc_v:<10.4f}")
    print(f"{'DD':<10} | {dd_v:<10.5f}")
    print(f"{'='*54}\n")

    # 防呆设计：检查波段数，如果是 PU 取 [29,19,9]，其他数据集取前三个
    bands = [29, 19, 9] if S_true.shape[2] >= 30 else [0, 1, 2]
    
    gt_rgb = matlab_style_rgb(S_true, bands)
    pred_rgb = matlab_style_rgb(Z_pred, bands)
    err_map = np.mean(np.abs(S_true - Z_pred), axis=2)
    
    plt.imsave(os.path.join(save_dir, f'GT_RGB_epoch_{epoch}.png'), gt_rgb)
    plt.imsave(os.path.join(save_dir, f'Recon_RGB_epoch_{epoch}.png'), pred_rgb)
    plt.imsave(os.path.join(save_dir, f'Error_Map_epoch_{epoch}.png'), err_map, cmap='jet', vmin=0, vmax=0.05)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1); plt.imshow(gt_rgb); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(pred_rgb); plt.title(f"Reconstruction (PSNR:{psnr_v:.2f})"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(err_map, cmap='jet', vmin=0, vmax=0.05); plt.title("Error Map"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Combined_Vis_epoch_{epoch}.png'), bbox_inches='tight')
    plt.close()








