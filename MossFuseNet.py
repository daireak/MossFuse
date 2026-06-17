import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from LKNet import UniRepLKNetBlock
from net import *
from degradation_net import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Encoder_MSI(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=32,
                 dim=64,
                 num_blocks=4,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Encoder_MSI, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.baseFeature = nn.Sequential(UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
                                         )
        self.detailFeature = nn.Sequential(UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False)
                                         )
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Encoder_HSI(nn.Module):
    def __init__(self,
                 inp_channels=31,
                 out_channels=32,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Encoder_HSI, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.baseFeature = nn.Sequential(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                         UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False)
                                         )
        self.detailFeature = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class Decoder_MSI(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=3,
                 dim=64,
                 num_blocks=4,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder_MSI, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.fuse = Spatial_Aware_Aggregation(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks-1)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = self.fuse(base_feature, detail_feature)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level0, out_enc_level1


class Decoder_HSI(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=31,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder_HSI, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.fuse = Spectral_Aware_Aggregation(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks-1)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = self.fuse(base_feature, detail_feature)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level0, out_enc_level1


class Decoder_LR(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=3,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder_LR, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level1


class Decoder_HR(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=31,
                 dim=64,
                 num_blocks=[2],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder_HR, self).__init__()
        self.fuse = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.fuse1 = Spatial_Aware_Aggregation(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.fuse2 = Spectral_Aware_Aggregation(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
    def forward(self, base_feature1, base_feature2, detail_feature_spatial, detail_feature_spectral):
        base_feature = self.fuse(torch.concat([base_feature1, base_feature2],1))
        out_enc_level0 = self.fuse1(base_feature, detail_feature_spatial)
        out_enc_level0 = self.fuse2(out_enc_level0, detail_feature_spectral)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level1


class MossFuse(nn.Module):
    # 【改动重点】这里加入 channels_MSI 和 channels_HSI 参数，默认值设为 3 和 31 以保持向后兼容
    def __init__(self, dim=48, num_blocks=1, scale=32, channels_MSI=3, channels_HSI=31):
        super(MossFuse, self).__init__()
        self.scale = scale
        self.channels_MSI = channels_MSI
        self.channels_HSI = channels_HSI
        
        # 将通道参数向下传递给各个编解码器
        self.modelE_MSI = Encoder_MSI(inp_channels=channels_MSI, dim=dim, num_blocks=num_blocks)   
        self.modelE_HSI = Encoder_HSI(inp_channels=channels_HSI, dim=dim, num_blocks=num_blocks)  
        self.modelD_MSI = Decoder_MSI(out_channels=channels_MSI, dim=dim, num_blocks=num_blocks)   
        self.modelD_HSI = Decoder_HSI(out_channels=channels_HSI, dim=dim, num_blocks=num_blocks)   
        
        # 将通道传递给退化网络（这要求你在 degradation_net.py 里也做对应的接口接收修改，请把那个文件发给我！）
        self.blind = BlindNet(ker_size=7, ratio=scale, channels_MSI=channels_MSI, channels_HSI=channels_HSI)    
        
        self.modelD_LR = Decoder_LR(out_channels=channels_MSI, dim=dim, num_blocks=num_blocks)      
        self.modelD_HR = Decoder_HR(out_channels=channels_HSI, dim=dim, num_blocks=num_blocks)      
        
        # 如果这两个类也硬编码了，同样需要传递通道数
        self.spatial_down = BlurDown(ratio=scale)
        self.spectral_down = SRF_Down()

    def forward(self, MSI, HSI):
        HSI_ = torch.nn.functional.interpolate(HSI, scale_factor=(self.scale, self.scale), mode='bilinear')
        RGB_E = self.modelE_MSI(MSI)
        HSI_E = self.modelE_HSI(HSI_)
        _, RGB_D = self.modelD_MSI(RGB_E[0], RGB_E[1])
        _, HSI_D = self.modelD_HSI(HSI_E[0], HSI_E[1])
        lr_msi_fhsi, lr_msi_fmsi, srf, psf = self.blind(HSI, MSI)
        LR_D = self.modelD_LR(RGB_E[0], HSI_E[0])
        HR_HSI = self.modelD_HR(RGB_E[0], HSI_E[0], RGB_E[1], HSI_E[1])
        msi_fhsi = self.spectral_down(HR_HSI, srf)
        hsi_fhsi = self.spatial_down(HR_HSI, psf)

        return RGB_E[0], RGB_E[1], HSI_E[0], HSI_E[1], RGB_D, HSI_D, lr_msi_fhsi, lr_msi_fmsi, LR_D, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

if __name__ == '__main__':
    # 模拟 PU 数据集的设置 (缩放倍数4，多光谱4波段，高光谱103波段)
    test_scale = 4
    test_msi_channels = 4
    test_hsi_channels = 103
    height = 128
    width = 128
    
    input_tensor1 = torch.rand(2, test_msi_channels, height, width).cuda()
    # 注意这里 HSI 的大小应该是 height/scale
    input_tensor2 = torch.rand(2, test_hsi_channels, height//test_scale, width//test_scale).cuda()
    
    # 动态初始化模型
    model = MossFuse(scale=test_scale, channels_MSI=test_msi_channels, channels_HSI=test_hsi_channels).cuda()
    
    with torch.no_grad():
        RGB_E0, EGB_E1, HSI_E0, HSI_E1, RGB_D, HSI_D, lr_msi_fhsi, lr_msi_fmsi, LR_D, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(input_tensor1, input_tensor2)
    print("Output HR_HSI size:", HR_HSI.size())
    print('Parameter number of MossFuse is ', sum(param.numel() for param in model.parameters()))
    print("PyTorch version:", torch.__version__)

    # 遍历模型的每个模块并打印参数量
    used_parameter_in_testing = 0
    for name, module in model.named_modules():
        if name != '':  # 排除根模块
            params = count_parameters(module)
            if name in ['modelE_MSI', 'modelE_HSI', 'blind', 'modelD_HR', 'spatial_down', 'spectral_down']:
                print(f"Module: {name}, Parameters: {params}")
                used_parameter_in_testing += params
    print("Total parameter number for inference: ", used_parameter_in_testing)