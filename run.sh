#!/bin/bash
#SBATCH --job-name=MossFuse_Train
#SBATCH --output=logs/mossfuse_%j.log
#SBATCH --error=logs/mossfuse_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

# 激活您的虚拟环境 (保持与您上传的 run.sh 一致)
source /home/dengxiaogui/NTSR/venv/bin/activate

# 创建日志和模型权重保存文件夹
mkdir -p logs
mkdir -p models/pth

echo "=========================================================="
echo "Starting MossFuse Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# ================= 实验控制区 =================
# 想要跑哪个数据集，就把对应的注释(#)删掉，并给另外两个加上注释即可。

# 【1】PU 数据集
# 设置: 波段93, 裁剪256, 缩放4倍
# python train.py --dataset_type single \
#                 --data_path ./dataset/PU/PU.mat \
#                 --r_path ./dataset/PU/R.mat \
#                 --mat_key paviaU \
#                 --r_key R \
#                 --scale 4 \
#                 --channels_hsi 93 \
#                 --channels_msi 4 \
#                 --batch_size 1 \
#                 --patch_size 256 \
#                 --epochs 1000 \
#                 --lr 1e-3 \
#                 --exp_name PU_Exp
python train.py --dataset_type single \
                --data_path //home/dengxiaogui/Data/PU.mat \
                --r_path //home/dengxiaogui/Data/R.mat \
                --mat_key img \
                --r_key R \
                --scale 4 \
                --channels_hsi 93 \
                --channels_msi 4 \
                --batch_size 1 \
                --patch_size 256 \
                --epochs 60 \
                --lr 1e-4 \
                --gpu 1 \
                --exp_name PU_Exp

# 【2】WDC 数据集
# 设置: 波段103, 裁剪256, 缩放4倍
#python train.py --dataset_type single --data_path ./dataset/WDC/WDC.mat --r_path ./dataset/WDC/R.mat --mat_key wdc --r_key R --scale 4 --channels_hsi 103 --channels_msi 4 --batch_size 1 --patch_size 256 --epochs 1000 --lr 1e-3 --exp_name WDC_Exp


# 【3】Chikusei 数据集
# 设置: 波段103, 裁剪512, 缩放8倍
# 注意: 因为裁剪尺寸达到了512，为防止显存溢出(OOM)，已将 batch_size 默认调小至 2 或 4
# python train.py --dataset_type single \
#                 --data_path ./dataset/Chikusei/Chikusei.mat \
#                 --r_path ./dataset/Chikusei/R.mat \
#                 --mat_key chikusei \
#                 --r_key R \
#                 --scale 8 \
#                 --channels_hsi 103 \
#                 --channels_msi 4 \
#                 --batch_size 1 \
#                 --patch_size 512 \
#                 --epochs 1000 \
#                 --lr 1e-3 \
#                 --exp_name Chikusei_Exp

echo "Job finished."