"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

"""
fa-vae model definition
"""

import torch
import torch.nn as nn

from .codec import *
from .discriminator import Discriminator, PatchDiscriminator

# VQGAN with Frequency Complement Module
class VQGANFCM(nn.Module):

    def _get_gaussian_kernel1d(self, kernel_size, sigma, device):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d


    def _get_gaussian_kernel2d(self, kernel_size, sigma, dtype, device):
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size, sigma, device).to(device)
        kernel1d_y = kernel1d_y.unsqueeze(1)
        kernel2d = torch.matmul(kernel1d_y, kernel1d_y.t())
        return kernel2d

    def _gaussian_blur(self, x, i, device):
        self.gauss_kernel = self._get_gaussian_kernel2d(self.kernel_size, self.sigmas[i], dtype=torch.float, device=device)
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-3],1,1,1)
        x = F.pad(x, self.padding, mode="reflect")
        feat = F.conv2d(x, self.gauss_kernel, groups=x.shape[-3])

        return feat


    def __init__(self, codebook_size, n_embed, double_z=False, ch_mult=(1,2,4,8), attn_resolutions=[], use_cosine_sim=False, codebook_dim=None, 
                    orthogonal_reg_weight=0, orthogonal_reg_max_codes=None, orthogonal_reg_active_codes_only=False, 
                    use_l2_quantizer=False, sync_codebook=False, commitment_weight=1.0, kernel_size=0, dsl_init_sigma=None,
                    use_non_pair_conv=False, device=None, use_gauss_resblock=False, use_gauss_attn=False, use_same_conv_gauss=False, use_same_gauss_resblock=False,
                    use_ffl_with_fcm=False, inference=False, num_groups=32, use_patch_discriminator=False, disc_n_layers=None):
        super().__init__()
        self.inference = inference

        if use_same_conv_gauss or use_same_gauss_resblock:
            self.use_same_gauss = True
        else:
            self.use_same_gauss = False
        self.device = device
        
        if use_non_pair_conv:
            print("\nMODEL SETTING: using non-pairwise DSL and convolution FCM....")
            self.gauss_kernels = None

            self.encoder = EncoderGauss(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)
            self.decoder = DecoderFcmGauss(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)
        
        elif use_same_conv_gauss:
            print("\nMODEL SETTING: using pair-wise DSL and convolution FCM...")
            self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True)
            self.kernel_size = kernel_size
            self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

            self.encoder = Encoder(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions)
            self.decoder = DecoderFcmGaussSame(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, device=device, num_groups=num_groups)

        elif use_same_gauss_resblock:
            print("\nMODEL SETTING: using pair-wise DSL and residual FCM...")
            self.sigmas = nn.Parameter(torch.tensor([dsl_init_sigma, dsl_init_sigma, dsl_init_sigma, dsl_init_sigma]), requires_grad=True)
            self.kernel_size = kernel_size
            self.padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

            self.encoder = Encoder(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions)
            self.decoder = DecoderFcmGaussSameResblock(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, device=device)

        elif use_gauss_resblock:
            print("\nMODEL SETTING: using non-pairwise DSL and residual FCM....")
            self.encoder = EncoderGauss(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)
            self.decoder = DecoderFcmResGauss(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)

        elif use_gauss_attn:
            print("\nMODEL SETTING: using non-pairwise DSL and attention FCM....")
            self.encoder = EncoderGauss(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)
            self.decoder = DecoderFcmAttnGauss(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions, kernel_size=kernel_size, dsl_init_sigma=dsl_init_sigma, device=device)
            
        elif use_ffl_with_fcm:
            print("\nMODEL SETTING: using convolution FCM....")
            self.encoder = Encoder(z_channels=n_embed, double_z=double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions)
            self.decoder = DecoderFcm(z_channels=n_embed, ch_mult=ch_mult, attn_resolutions=attn_resolutions)
            
        self.use_l2_quantizer = use_l2_quantizer

        if use_l2_quantizer:
            print("\nMODEL SETTING: using L2 regularization in the quantizer. With distributed training? {}".format(sync_codebook))
            from .l2_quantize import VectorQuantize
            self.quantizer = VectorQuantize(codebook_size=codebook_size, dim=n_embed, accept_image_fmap = True, use_cosine_sim=use_cosine_sim, codebook_dim=codebook_dim,
                       orthogonal_reg_weight=orthogonal_reg_weight, orthogonal_reg_max_codes=orthogonal_reg_max_codes, orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
                       sync_codebook=sync_codebook, commitment_weight=commitment_weight)

        if use_patch_discriminator:
            self.discriminator = PatchDiscriminator(n_layers=disc_n_layers)
        else:
            self.discriminator = Discriminator()

    def encode(self, x):
        z, enc_feats = self.encoder(x, inference=self.inference)  # map to latent space, [N, C, H, W]
        if self.use_l2_quantizer:
            z_q, indices, loss_q = self.quantizer(z)  # quantize
        else:
            z_q, loss_q, indices = self.quantizer(z)  # quantize
        return z_q, loss_q, indices, enc_feats

    def decode(self, z):
        x_recon, dec_feats = self.decoder(z, inference=self.inference)
        return x_recon, dec_feats

    def forward(self, x, stage=0, inference=False):
        if stage == 0:
            # Stage 0: training E + G + Q
            z, loss_q, _, enc_feats = self.encode(x)
            x_recon, dec_feats = self.decode(z)
            logits_fake = self.discriminator(x_recon)

            if self.use_same_gauss and not inference:
                for i in range(len(enc_feats)):
                    enc_feats[i] = self._gaussian_blur(enc_feats[i], i, device=self.device)
                    dec_feats[3-i] = self._gaussian_blur(dec_feats[3-i], 3-i, device=self.device)

            return x_recon, loss_q, logits_fake, z, enc_feats, dec_feats

        elif stage == 1:
            # Stage 1: training D
            with torch.no_grad():
                z, loss_q, _, _ = self.encode(x)
                x_recon, _ = self.decode(z)

            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())
            return logits_real, logits_fake

        else:
            raise ValueError(f"Invalid stage: {stage}")
