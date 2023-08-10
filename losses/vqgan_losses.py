"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

import torch
import torchvision.transforms as T

torch.manual_seed(0)


def recon_ffl_loss(ffl, x, x_recon):
    return ffl(x_recon, x)

# this loss is to penalize the difference between the features, en_feat and de_feat
# suitable for outputs of encoder and decoder blocks, also for DSL
def recon_ffl_features_loss(ffl, en_feat, de_feat, device):

    de_feat.reverse()

    loss = torch.tensor([0.], requires_grad = True, device=device)
    losses = []
    for i in range(len(en_feat)):
        loss = loss + ffl(de_feat[i], en_feat[i])
        losses.append(ffl(de_feat[i], en_feat[i]))

    loss = loss / len(en_feat)
    
    return loss, losses


# Spectrum Loss (SL), with deterministic sigmas
def recon_sl_gaussian_features_loss(ffl, gaussian_kernel, gaussian_sigma, en_feat, de_feat, device):
    blurrer = T.GaussianBlur(kernel_size=(gaussian_kernel, gaussian_kernel), sigma=gaussian_sigma)
    
    de_feat.reverse()
    loss = torch.tensor([0.], requires_grad = True, device=device)
    losses = []

    en_feat = [blurrer(feat) for feat in en_feat]
    de_feat = [blurrer(feat) for feat in de_feat]

    for i in range(len(en_feat)):
        loss = loss + ffl(de_feat[i], en_feat[i])
        losses.append(ffl(de_feat[i], en_feat[i]))

    loss = loss / len(en_feat)
    
    return loss, losses
