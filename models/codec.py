"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, value=0)
        x = self.conv(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        )

        self.has_shortcut = (in_c != out_c)
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.block(x)
        if self.has_shortcut:
            x = self.shortcut(x)

        return x + h

#####################################################
####### The FCM with convolutional architecture #####
#####################################################
class NonResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout, num_groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, out_c),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        )

        self.has_shortcut = (in_c != out_c)
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.block(x)
        if self.has_shortcut:
            x = self.shortcut(x)

        return h


class AttnBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_c)
        self.attn = nn.MultiheadAttention(in_c, num_heads=1, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        out, _ = self.attn(h, h, h, need_weights=False)  # [B, H*W, C]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
        out = x + out

        return out


################################################
####### The FCM with attention architecture #####
################################################
class TransEncoderBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_c)
        self.attn = nn.TransformerEncoderLayer(in_c, nhead=8, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        out = self.attn(h)  # [B, H*W, C]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]

        return out

###### the encoder architecture #######
class Encoder(nn.Module):
    def __init__(
        self,
        in_c=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        double_z=True
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_c, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        for level in range(len(ch_mult)):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))

            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2

        self.down = nn.Sequential(*blocks)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(z_channels, z_channels, kernel_size=1),
        )

    def forward(self, x, inference=False):
        inter_features = []
        h = self.conv_in(x)           
        inter_features.append(h)
        h = self.down(h)                
        inter_features.append(h)
        h = self.mid(h)                
        inter_features.append(h)
        h = self.final(h)              
        inter_features.append(h)

        return h, inter_features


##### DSL with non-pairwise sigmas, 
##### thus, encoder and decoder together have 8 sigmas 
class EncoderGauss(nn.Module):
    def __init__(
        self,
        in_c=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        double_z=True,
        kernel_size=3,
        dsl_init_sigma=None,
        device=None

    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.device = device
        # initialize the gaussian kernel sigmas
        self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True) 
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.conv_in = nn.Conv2d(in_c, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        for level in range(len(ch_mult)):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))

            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2

        self.down = nn.Sequential(*blocks)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(z_channels, z_channels, kernel_size=1),
        )

    def _get_gaussian_kernel1d(self, kernel_size, sigma, device):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d


    def _get_gaussian_kernel2d(self, kernel_size, sigma, dtype, device):
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size, sigma, device)
        kernel1d_y = kernel1d_y.unsqueeze(1)
        kernel2d = torch.matmul(kernel1d_y, kernel1d_y.t())
        return kernel2d


    def _gaussian_blur(self, x, i):
        self.gauss_kernel = self._get_gaussian_kernel2d(self.kernel_size, self.sigmas[i], dtype=torch.float, device=self.device)
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-3],1,1,1)
        x = F.pad(x, self.padding, mode="reflect")
        feat = F.conv2d(x, self.gauss_kernel, groups=x.shape[-3])

        return feat


    def forward(self, x, inference=False):
        inter_features = []
        h = self.conv_in(x)          
        if not inference:
            feat = self._gaussian_blur(h, 0)
        else:
            feat = h

        inter_features.append(feat)

        h = self.down(h)               
        if not inference:
            feat = self._gaussian_blur(h, 1)
        else:
            feat = h

        inter_features.append(feat)

        h = self.mid(h)                 

        if not inference:
            feat = self._gaussian_blur(h, 2)
        else:
            feat = h
        inter_features.append(feat)

        h = self.final(h)               

        if not inference:
            feat = self._gaussian_blur(h, 3)
        else:
            feat = h
        inter_features.append(feat)

        return h, inter_features


##### DSL with pairwise sigmas, 
##### thus, encoder and decoder together have 4 sigmas
##### each level of encoder and decoder block shares the same sigma  
class EncoderGaussSameSigma(nn.Module):
    def __init__(
        self,
        in_c=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        double_z=True,
        kernel_size=3,
        device=None,
        sigmas=None

    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.sigmas = sigmas

        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.conv_in = nn.Conv2d(in_c, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        for level in range(len(ch_mult)):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))

            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2

        self.down = nn.Sequential(*blocks)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(z_channels, z_channels, kernel_size=1),
        )

    def forward(self, x):

        inter_features = []
        h = self.conv_in(x)           
        inter_features.append(h)

        h = self.down(h)                
        inter_features.append(h)

        h = self.mid(h)                
        inter_features.append(h)

        h = self.final(h)              
        inter_features.append(h)

        return h, inter_features


#### decoder with FCM without gaussian kernels
class Decoder(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
    ):
        super().__init__()

        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.quant_conv_in = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2

        self.up = nn.Sequential(*blocks)

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        inter_features = []
        h = self.quant_conv_in(z)      
        inter_features.append(h)
        h = self.conv_in(h)             
        inter_features.append(h)
        h = self.mid(h)                 
        inter_features.append(h)
        h = self.up(h)                  
        inter_features.append(h)
        h = self.final(h)               

        return h, inter_features


#######################################################
####### Decoder with convolution FCM architecture #####
#######################################################
class DecoderFcm(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
    ):
        super().__init__()
        print("\nDECODER SETTING: Using decoder with convolution FCM...")
        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.fcm_1 = NonResnetBlock(z_channels, z_channels, dropout=dropout)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = NonResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z, inference=False): 

        inter_features = []    
        h = self.fcm_1(z)               
        inter_features.append(h)
        h = h + z
        h_ = self.conv_in(h)             

        h = self.fcm_2(h_)               
        inter_features.append(h)
        h_ = h_ + h
        h_ = self.mid(h_)                 

        h = self.fcm_3(h_)               
        inter_features.append(h)
        h_ = h_ + h
        h_ = self.up(h_)                  

        h = self.fcm_4(h_)               
        inter_features.append(h)
        h_ = h_ + h
        h = self.final(h_)               

        return h, inter_features


#####################################################
####### FCM with convolution, DSL non pair-wise #####
#####################################################
class DecoderFcmGauss(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        kernel_size=0,
        dsl_init_sigma=None,
        device=None
    ):
        super().__init__()

        print("\nDECODER SETTING: with non pairwise DSL")
        self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True)
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.device = device
        self.kernel_size = kernel_size
        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        # self.quant_conv_in = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.fcm_1 = NonResnetBlock(z_channels, z_channels, dropout=dropout)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = NonResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )


    def _get_gaussian_kernel1d(self, kernel_size, sigma, device):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d


    def _get_gaussian_kernel2d(self, kernel_size, sigma, dtype, device):
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size, sigma, device)
        kernel1d_y = kernel1d_y.unsqueeze(1)
        kernel2d = torch.matmul(kernel1d_y, kernel1d_y.t())
        return kernel2d

    def _gaussian_blur(self, x, i):
        self.gauss_kernel = self._get_gaussian_kernel2d(self.kernel_size, self.sigmas[i], dtype=torch.float, device=self.device)
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-3],1,1,1)
        x = F.pad(x, self.padding, mode="reflect")
        feat = F.conv2d(x, self.gauss_kernel, groups=x.shape[-3])

        return feat



    def forward(self, z, inference=False): 
        feat = None
        inter_features = []    
        h = self.fcm_1(z)               
        if not inference:
            feat = self._gaussian_blur(h, 0)
        else:
            feat = h
        
        inter_features.append(feat)
        h = h + z

        h_ = self.conv_in(h)             
        h = self.fcm_2(h_)               

        if not inference:
            feat = self._gaussian_blur(h, 1)
        else:
            feat = h
        inter_features.append(feat)
        h_ = h_ + h

        h_ = self.mid(h_)                 
        h = self.fcm_3(h_)              

        if not inference:
            feat = self._gaussian_blur(h, 2)
        else:
            feat = h
        inter_features.append(feat)
        h_ = h_ + h

        h_ = self.up(h_)                  
        h = self.fcm_4(h_)               

        if not inference:
            feat = self._gaussian_blur(h, 3)
        else:
            feat = h
        inter_features.append(feat)
        h_ = h_ + h
        
        h = self.final(h_)               

        return h, inter_features


#################################################
####### FCM with convolution, pair-wise DSL #####
#################################################
class DecoderFcmGaussSame(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        kernel_size=0,
        device=None,
        num_groups=32
    ):
        super().__init__()
        print("\nDECODER SETTING: using convolution decoder fcm with pair-wise DSL")
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.device = device
        self.kernel_size = kernel_size
        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.fcm_1 = NonResnetBlock(z_channels, z_channels, dropout=dropout, num_groups=num_groups)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = NonResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = NonResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z, inference=False): 

        inter_features = []    
        h = self.fcm_1(z)               
        inter_features.append(h)
        h = h + z

        h_ = self.conv_in(h)            
        h = self.fcm_2(h_)              
        inter_features.append(h)
        h_ = h_ + h

        h_ = self.mid(h_)                
        h = self.fcm_3(h_)             
        inter_features.append(h)
        h_ = h_ + h

        h_ = self.up(h_)                
        h = self.fcm_4(h_)              
        inter_features.append(h)
        h_ = h_ + h
        
        h = self.final(h_)             

        return h, inter_features


#########################################################
####### FCM with residual connection, pair-wise DSL #####
#########################################################
class DecoderFcmGaussSameResblock(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        kernel_size=0,
        device=None,
    ):
        super().__init__()
        print("\nDECODER SETTING: using pair-wise DSL with RES FCM.")
    
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.device = device
        self.kernel_size = kernel_size
        block_in = ch * ch_mult[len(ch_mult) - 1]

        self.fcm_1 = ResnetBlock(z_channels, z_channels, dropout=dropout)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = ResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = ResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = ResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, z, inference=False): 
        inter_features = []    
        h = self.fcm_1(z)              
        inter_features.append(h)

        h = self.conv_in(h)            
        h = self.fcm_2(h)               
        inter_features.append(h)

        h = self.mid(h)                
        h = self.fcm_3(h)              
        inter_features.append(h)

        h = self.up(h)                  
        h = self.fcm_4(h)              
        inter_features.append(h)
        
        h = self.final(h)               

        return h, inter_features


#############################################################
####### FCM with residual connection, non pair-wise DSL #####
#############################################################
class DecoderFcmResGauss(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        kernel_size=0,
        dsl_init_sigma=None, 
        device=None
    ):
        super().__init__()
        self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True)
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.device = device
        self.kernel_size = kernel_size
        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.fcm_1 = ResnetBlock(z_channels, z_channels, dropout=dropout)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = ResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = ResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = ResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )


    def _get_gaussian_kernel1d(self, kernel_size, sigma, device):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d


    def _get_gaussian_kernel2d(self, kernel_size, sigma, dtype, device):
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size, sigma, device)
        kernel1d_y = kernel1d_y.unsqueeze(1)
        kernel2d = torch.matmul(kernel1d_y, kernel1d_y.t())
        return kernel2d

    def _gaussian_blur(self, x, i):
        self.gauss_kernel = self._get_gaussian_kernel2d(self.kernel_size, self.sigmas[i], dtype=torch.float, device=self.device)
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-3],1,1,1)
        x = F.pad(x, self.padding, mode="reflect")
        feat = F.conv2d(x, self.gauss_kernel, groups=x.shape[-3])

        return feat



    def forward(self, z, inference=False):
        feat = None
        inter_features = []    
        h = self.fcm_1(z)              

        if not inference:
            feat = self._gaussian_blur(h, 0)
        inter_features.append(feat)

        h = self.conv_in(h)            
        h = self.fcm_2(h)             

        if not inference:
            feat = self._gaussian_blur(h, 1)
        inter_features.append(feat)

        h = self.mid(h)               
        h = self.fcm_3(h)              

        if not inference:
            feat = self._gaussian_blur(h, 2)
        inter_features.append(feat)

        h = self.up(h)                
        h = self.fcm_4(h)            

        if not inference:
            feat = self._gaussian_blur(h, 3)
        inter_features.append(feat)
        
        h = self.final(h)            

        return h, inter_features


###################################################
####### FCM with attention, non pair-wise DSL #####
####### the first three blocks are attention layers, the last one is residual  
###################################################
class DecoderFcmAttnGauss(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
        kernel_size=0,
        dsl_init_sigma=None,
        device=None
    ):
        super().__init__()
        self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True)
        self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        self.device = device
        self.kernel_size = kernel_size
        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.fcm_1 = TransEncoderBlock(z_channels)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = TransEncoderBlock(block_in)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = TransEncoderBlock(block_in)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = ResnetBlock(block_in, block_in, dropout=0.1)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )


    def _get_gaussian_kernel1d(self, kernel_size, sigma, device):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d


    def _get_gaussian_kernel2d(self, kernel_size, sigma, dtype, device):
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size, sigma, device)
        kernel1d_y = kernel1d_y.unsqueeze(1)
        kernel2d = torch.matmul(kernel1d_y, kernel1d_y.t())
        return kernel2d

    def _gaussian_blur(self, x, i):
        self.gauss_kernel = self._get_gaussian_kernel2d(self.kernel_size, self.sigmas[i], dtype=torch.float, device=self.device)
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-3],1,1,1)
        x = F.pad(x, self.padding, mode="reflect")
        feat = F.conv2d(x, self.gauss_kernel, groups=x.shape[-3])

        return feat


    def forward(self, z, inference=False):
        feat = None
        inter_features = []    
        h = self.fcm_1(z)               
        if not inference:
            feat = self._gaussian_blur(h, 0)
        inter_features.append(feat)

        h = self.conv_in(h)            
        h = self.fcm_2(h)              
        if not inference:
            feat = self._gaussian_blur(h, 1)
        inter_features.append(feat)

        h = self.mid(h)               
        h = self.fcm_3(h)              
        if not inference:
            feat = self._gaussian_blur(h, 2)
        inter_features.append(feat)

        h = self.up(h)                
        h = self.fcm_4(h)             
        if not inference:
            feat = self._gaussian_blur(h, 3)
        inter_features.append(feat)
        
        h = self.final(h)              

        return h, inter_features


### original decoder from clip-gen ###
class DecoderFcmOld(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
    ):
        super().__init__()

        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.fcm_1 = ResnetBlock(z_channels, z_channels, dropout=dropout)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.fcm_2 = ResnetBlock(block_in, block_in, dropout=dropout)
        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        self.fcm_3 = ResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2 

        self.up = nn.Sequential(*blocks)

        self.fcm_4 = ResnetBlock(block_in, block_in, dropout=dropout)
        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z): # z.shape = (15,256,16,16)
        self.inter_features = []    
        h = self.fcm_1(z)               # h.shape = (15,256,16,16)
        self.inter_features.append(h)
        h_ = self.conv_in(h)             # h.shape = (15,512,16,16)

        h = self.fcm_2(h_)               # h.shape = (15,512,16,16)
        self.inter_features.append(h)
        h_ = self.mid(h)                 # h.shape = (15,512,16,16)

        h = self.fcm_3(h_)               # h.shape = (15,512,16,16)
        self.inter_features.append(h)
        h_ = self.up(h)                  # h.shape = (15,128,256,256)

        h = self.fcm_4(h_)               # h.shape = (15,128,256,256)
        self.inter_features.append(h)
        h = self.final(h)               # h.shape = (15,3,256,256)

        return h
