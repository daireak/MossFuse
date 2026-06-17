import random
import numpy as np
import torch
import torch.utils.data as udata
import os
import hdf5storage as hdf5
import scipy.io as scio
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================================================
# 1. 针对多图数据集 (如 CAVE, Harvard) 的数据加载器
# =========================================================================
class MultiImageDatasetTrain(udata.Dataset):
    def __init__(self, data_dir, r_path, file_list, mat_key='rad', r_key='resp', scale=32, patch_size=128, stride=64):
        self.data_dir = data_dir
        self.r_path = r_path
        self.keys = file_list
        self.mat_key = mat_key
        self.r_key = r_key
        self.scale = scale
        self.patch_size = patch_size
        self.stride = stride

        self.msi_list = []
        self.hsi_g_list = []
        self.hsi_list = []

        # 读取光谱响应矩阵 (SRF)
        try:
            res = hdf5.loadmat(self.r_path)[self.r_key]
        except:
            res = scio.loadmat(self.r_path)[self.r_key]
            
        # 确保 res 的形状为 (channels_HSI, channels_MSI)
        if res.shape[0] < res.shape[1]:  
            res = np.transpose(res, (1, 0))

        for i in range(len(self.keys)):
            mat_path = os.path.join(self.data_dir, self.keys[i])
            try:
                mat = hdf5.loadmat(mat_path)
                hyper = np.float32(np.array(mat[self.mat_key]) / (2**16 - 1)) # CAVE默认使用了 16bit 归一化
            except:
                mat = scio.loadmat(mat_path)
                hyper = np.float32(np.array(mat[self.mat_key]))
                if hyper.max() > 1.0:
                    hyper = hyper / hyper.max() # 通用归一化

            # 生成 LR_HSI (空间下采样)
            hyper1 = cv2.GaussianBlur(hyper, ((scale-1), (scale-1)), 2)
            hyper1 = hyper1[(scale//2-1)::scale, (scale//2-1)::scale, :]
            
            # 生成 HR_MSI (光谱下采样)
            rgb = np.tensordot(hyper, res, axes=([2], [0]))

            self.hsi_g_list.append(hyper)
            self.hsi_list.append(hyper1)
            self.msi_list.append(rgb)
        
        # 计算每张图能切多少个 patch
        h, w, _ = self.hsi_g_list[0].shape
        self.num_row_patches = (h - self.patch_size) // self.stride + 1
        self.num_col_patches = (w - self.patch_size) // self.stride + 1
        self.num_patches_per_img = self.num_row_patches * self.num_col_patches

        print(f'{len(self.keys)} train image pairs loaded! Total patches: {len(self.keys) * self.num_patches_per_img}')

    def __len__(self):
        return len(self.keys) * self.num_patches_per_img

    def __getitem__(self, index):
        index_img = index // self.num_patches_per_img
        index_inside_image = index % self.num_patches_per_img
        index_row = index_inside_image // self.num_col_patches
        index_col = index_inside_image % self.num_col_patches

        h_start = index_row * self.stride
        w_start = index_col * self.stride
        
        lr_h_start = h_start // self.scale
        lr_w_start = w_start // self.scale
        lr_patch_size = self.patch_size // self.scale

        hsi_g = self.hsi_g_list[index_img][h_start:h_start+self.patch_size, w_start:w_start+self.patch_size, :]
        msi = self.msi_list[index_img][h_start:h_start+self.patch_size, w_start:w_start+self.patch_size, :]
        hsi = self.hsi_list[index_img][lr_h_start:lr_h_start+lr_patch_size, lr_w_start:lr_w_start+lr_patch_size, :]
        
        # 数据增强
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        for j in range(rotTimes):
            hsi_g = np.rot90(hsi_g)
            hsi = np.rot90(hsi)
            msi = np.rot90(msi)
        if vFlip:
            hsi_g = np.flip(hsi_g, axis=1)
            hsi = np.flip(hsi, axis=1)
            msi = np.flip(msi, axis=1)
        if hFlip:
            hsi_g = np.flip(hsi_g, axis=0)
            hsi = np.flip(hsi, axis=0)
            msi = np.flip(msi, axis=0)

        hsi = torch.Tensor(np.transpose(hsi.copy(), (2, 0, 1)))
        msi = torch.Tensor(np.transpose(msi.copy(), (2, 0, 1)))
        hsi_g = torch.Tensor(np.transpose(hsi_g.copy(), (2, 0, 1)))

        return msi, hsi, hsi_g


class MultiImageDatasetTest(udata.Dataset):
    def __init__(self, data_dir, r_path, file_list, mat_key='rad', r_key='resp', scale=32):
        self.data_dir = data_dir
        self.r_path = r_path
        self.keys = file_list
        self.mat_key = mat_key
        self.r_key = r_key
        self.scale = scale

        try:
            res = hdf5.loadmat(self.r_path)[self.r_key]
        except:
            res = scio.loadmat(self.r_path)[self.r_key]
            
        if res.shape[0] < res.shape[1]:
            res = np.transpose(res, (1, 0))

        self.hyper_list = []
        self.hyper1_list = []
        self.rgb_list = []
        for i in range(len(self.keys)):
            mat_path = os.path.join(self.data_dir, self.keys[i])
            try:
                mat = hdf5.loadmat(mat_path)
                hyper = np.float32(np.array(mat[self.mat_key]) / (2**16 - 1))
            except:
                mat = scio.loadmat(mat_path)
                hyper = np.float32(np.array(mat[self.mat_key]))
                if hyper.max() > 1.0:
                    hyper = hyper / hyper.max()

            hyper1 = cv2.GaussianBlur(hyper, ((scale-1), (scale-1)), 2)[(scale//2-1)::scale, (scale//2-1)::scale, :]
            rgb = np.tensordot(hyper, res, axes=([2], [0]))
            
            hyper1 = np.transpose(hyper1, [2, 0, 1])
            hyper = np.transpose(hyper, [2, 0, 1])
            rgb = np.transpose(rgb, [2, 0, 1])
            
            self.hyper_list.append(torch.Tensor(hyper))
            self.hyper1_list.append(torch.Tensor(hyper1))
            self.rgb_list.append(torch.Tensor(rgb))
        print(f'{len(self.keys)} test image pairs loaded!')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        rgb = self.rgb_list[index]
        hyper1 = self.hyper1_list[index]
        hyper = self.hyper_list[index]
        img_name = self.keys[index]
        return rgb, hyper1, hyper, img_name


# =========================================================================
# 2. 针对单张大图数据集 (如 PU, Houston) 的数据加载器
# =========================================================================
class SingleImageDatasetTrain(udata.Dataset):
    def __init__(self, data_path, r_path, mat_key='paviaU', r_key='R', scale=4, patch_size=64, num_patches=50):
        self.data_path = data_path
        self.r_path = r_path
        self.mat_key = mat_key
        self.r_key = r_key
        self.scale = scale
        self.patch_size = patch_size
        self.num_patches = num_patches

        try:
            mat = scio.loadmat(self.data_path)
            res = scio.loadmat(self.r_path)
        except:
            mat = hdf5.loadmat(self.data_path)
            res = hdf5.loadmat(self.r_path)
      
        self.hsi_img = np.float32(mat[self.mat_key])
        self.hsi_img = self.hsi_img[:256, :256, :]
        self.R = np.float32(res[self.r_key])

        if self.hsi_img.max() > 1.0:
            self.hsi_img = self.hsi_img / self.hsi_img.max()

        if self.R.shape[0] < self.R.shape[1]:
            self.R = np.transpose(self.R, (1, 0))

        # ================== 核心修复：自动对齐波段维度 ==================
        hsi_bands = self.hsi_img.shape[-1]
        r_bands = self.R.shape[0]
        
        if r_bands > hsi_bands:
            diff = r_bands - hsi_bands
            print(f"⚠️ R矩阵波段({r_bands}) > HSI波段({hsi_bands})，自动舍弃 R 的【末尾】 {diff} 个波段！")
            self.R = self.R[:-diff, :]  # <-- 改为了 [:-diff, :]
        elif hsi_bands > r_bands:
            diff = hsi_bands - r_bands
            print(f"⚠️ HSI波段({hsi_bands}) > R矩阵波段({r_bands})，自动舍弃 HSI 的【末尾】 {diff} 个波段！")
            self.hsi_img = self.hsi_img[:, :, :-diff] # <-- 改为了 [:, :, :-diff]
        # =========================================================================
        

        # ========== 终极修复 1：归一化 R 矩阵，保证 MSI 严格在 [0, 1] 之间 ==========
        col_sum = np.sum(self.R, axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        self.R = self.R / col_sum
        # ====================================================================

        self.msi_img = np.tensordot(self.hsi_img, self.R, axes=([2], [0]))
        
        print(f'Train single image loaded. HSI shape: {self.hsi_img.shape}, MSI shape: {self.msi_img.shape}')

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index):
        h, w, _ = self.hsi_img.shape
        
        # 随机裁剪，确保裁剪区域大小能被 scale 整除
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        hr_hsi = self.hsi_img[top:top+self.patch_size, left:left+self.patch_size, :]
        hr_msi = self.msi_img[top:top+self.patch_size, left:left+self.patch_size, :]
        
        # 生成 LR_HSI (空间下采样)
        lr_hsi = cv2.GaussianBlur(hr_hsi, (7, 7), 2)
        
        
        lr_hsi = lr_hsi[0::self.scale, 0::self.scale, :]

        # 数据增强
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        for j in range(rotTimes):
            hr_hsi = np.rot90(hr_hsi)
            lr_hsi = np.rot90(lr_hsi)
            hr_msi = np.rot90(hr_msi)
        if vFlip:
            hr_hsi = np.flip(hr_hsi, axis=1)
            lr_hsi = np.flip(lr_hsi, axis=1)
            hr_msi = np.flip(hr_msi, axis=1)
        if hFlip:
            hr_hsi = np.flip(hr_hsi, axis=0)
            lr_hsi = np.flip(lr_hsi, axis=0)
            hr_msi = np.flip(hr_msi, axis=0)

        # HWC to CHW
        hr_hsi = torch.Tensor(np.transpose(hr_hsi.copy(), (2, 0, 1)))
        hr_msi = torch.Tensor(np.transpose(hr_msi.copy(), (2, 0, 1)))
        lr_hsi = torch.Tensor(np.transpose(lr_hsi.copy(), (2, 0, 1)))
        
        return hr_msi, lr_hsi, hr_hsi


class SingleImageDatasetTest(udata.Dataset):
    def __init__(self, data_path, r_path, mat_key='paviaU', r_key='R', scale=4):
        self.data_path = data_path
        self.r_path = r_path
        self.mat_key = mat_key
        self.r_key = r_key
        self.scale = scale

        try:
            mat = scio.loadmat(self.data_path)
            res = scio.loadmat(self.r_path)
        except:
            mat = hdf5.loadmat(self.data_path)
            res = hdf5.loadmat(self.r_path)

        self.hsi_img = np.float32(mat[self.mat_key])
        self.hsi_img = self.hsi_img[:256, :256, :]
        self.R = np.float32(res[self.r_key])

        if self.hsi_img.max() > 1.0:
            self.hsi_img = self.hsi_img / self.hsi_img.max()

        if self.R.shape[0] < self.R.shape[1]:
            self.R = np.transpose(self.R, (1, 0))

        # ================== 核心修复：自动对齐波段维度 ==================
        hsi_bands = self.hsi_img.shape[-1]
        r_bands = self.R.shape[0]
        
        if r_bands > hsi_bands:
            diff = r_bands - hsi_bands
            print(f"⚠️ R矩阵波段({r_bands}) > HSI波段({hsi_bands})，自动舍弃 R 的【末尾】 {diff} 个波段！")
            self.R = self.R[:-diff, :]  # <-- 改为了 [:-diff, :]
        elif hsi_bands > r_bands:
            diff = hsi_bands - r_bands
            print(f"⚠️ HSI波段({hsi_bands}) > R矩阵波段({r_bands})，自动舍弃 HSI 的【末尾】 {diff} 个波段！")
            self.hsi_img = self.hsi_img[:, :, :-diff] # <-- 改为了 [:, :, :-diff]
        # =========================================================================
        # ========== 终极修复 1：归一化 R 矩阵，保证 MSI 严格在 [0, 1] 之间 ==========
        col_sum = np.sum(self.R, axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        self.R = self.R / col_sum
        # ====================================================================

        self.msi_img = np.tensordot(self.hsi_img, self.R, axes=([2], [0]))
        
        h, w, _ = self.hsi_img.shape
        h_new = h - (h % scale)
        w_new = w - (w % scale)
        
        self.hsi_img = self.hsi_img[:h_new, :w_new, :]
        self.msi_img = self.msi_img[:h_new, :w_new, :]

        self.lr_hsi = cv2.GaussianBlur(self.hsi_img, (7, 7), 2)
        self.lr_hsi = self.lr_hsi[0::self.scale, 0::self.scale, :]

    def __len__(self):
        return 1 # 单图测试，整个数据集只有一张全图

    def __getitem__(self, index):
        hr_hsi = torch.Tensor(np.transpose(self.hsi_img, (2, 0, 1)))
        hr_msi = torch.Tensor(np.transpose(self.msi_img, (2, 0, 1)))
        lr_hsi = torch.Tensor(np.transpose(self.lr_hsi, (2, 0, 1)))
        img_name = os.path.basename(self.data_path)
        
        # 严格保持测试接口的四个返回值输出：msi, lr_hsi, hr_hsi, img_name
        return hr_msi, lr_hsi, hr_hsi, img_name