
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from PatchGAN discriminator as in Pix2Pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park. All rights reserved.
# ------------------------------------------------------------------------------------
# Modified from Clip-Gen (https://github.com/HFAiLab/clip-gen/)
# Copyright (c) 2022 HFAiLab
# ------------------------------------------------------------------------------------

import torch.nn as nn
from math import log2, sqrt
from functools import partial
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = - logits_fake.mean() * 2 if logits_real is None else F.relu(1. + logits_fake).mean() 
    loss_real = 0 if logits_real is None else F.relu(1. - logits_real).mean()
    
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = F.softplus(-logits_fake).mean() * 2 if logits_real is None else F.softplus(logits_fake).mean()
    loss_real = 0 if logits_real is None else F.softplus(-logits_real).mean()
    
    return 0.5 * (loss_real + loss_fake)


def least_square_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = logits_fake.pow(2).mean() * 2 if logits_real is None else (1 + logits_fake).pow(2).mean()
    loss_real = 0 if logits_real is None else (1 - logits_real).pow(2).mean() 
    
    return 0.5 * (loss_real + loss_fake)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features: int,
                 logdet: Optional[bool] = False,
                 affine: Optional[bool] = True,
                 allow_reverse_init: Optional[bool] = False) -> None:
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: torch.FloatTensor) -> None:
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input: torch.FloatTensor, reverse: Optional[bool] = False) -> Union[torch.FloatTensor, Tuple]:
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output: torch.FloatTensor) -> torch.FloatTensor:
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
            
        return h


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, num_layers: int = 3, use_actnorm: bool = False) -> None:
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            num_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """Standard forward."""
        return self.main(input)


class Discriminator(nn.Module):

    def __init__(self, in_channel=3, channel=64, num_layer=3):
        super().__init__()

        modules = [nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1)]
        modules += [nn.LeakyReLU(0.2, True)]

        chs = [channel * min(2 ** i, 8) for i in range(num_layer + 1)]

        # increase channels
        for i in range(1, num_layer + 1):
            stride = 2 if i != num_layer else 1
            modules += [
                nn.Conv2d(chs[i - 1], chs[i], kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(chs[i]),
                nn.LeakyReLU(0.2, True)
            ]

        self.features = nn.Sequential(*modules)
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        out = self.head(x)
        return out
