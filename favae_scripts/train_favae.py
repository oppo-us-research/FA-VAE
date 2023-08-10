"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""


import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import yaml

import sys
sys.path.append(".")
sys.path.append("..")

from models.vqgan_fcm import VQGANFCM
from datasets.general_dataloader import load_data
from datasets.statistic import mean, std
from losses.lpips import LPIPS
from losses.hinge import hinge_d_loss, hinge_g_loss
from losses.vqgan_losses import *
from utils import save_model
from accelerate import Accelerator, DistributedDataParallelKwargs
from focal_frequency_loss import FocalFrequencyLoss as FFL
find_unused_parameters=True

torch.autograd.set_detect_anomaly(True)

def compute_adaptive_weight(model, loss_recon, loss_disc, accelerator):
    last_layer = accelerator.unwrap_model(model).decoder.final[2].weight
    grad_disc = torch.autograd.grad(loss_disc, last_layer, retain_graph=True)[0]
    grad_recon = torch.autograd.grad(loss_recon, last_layer, retain_graph=True)[0]

    weight_d = torch.norm(grad_recon) / (torch.norm(grad_disc) + 1e-4)
    weight_d = torch.clamp(weight_d, 0.0, 1e4).item()
    return weight_d


@torch.no_grad()
def log_recons_image(name, x, x_recon, steps, writer):
    std1 = torch.tensor(std).view(1, -1, 1, 1).cuda()
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).cuda()
    x_recon = x_recon * std1 + mean1  # [B, C, H, W]
    x = x * std1 + mean1

    img = torch.cat([x, x_recon], dim=0).clamp(0, 1)
    img = make_grid(img, x.size(0))

    writer.add_image(name, img, steps)
    writer.flush()


def train(loader, model, lpips, opt_g, opt_d, epoch, start_step, writer, accelerator, args, ffl_func=None, dsl_feature_func=None, sl_feature_func=None):
    model.train()
    device = accelerator.device
    steps_per_epoch = len(loader) + start_step

    # initialize loss values
    loss_ffl = torch.zeros(1).to(device)
    loss_dsl_features = torch.zeros(1).to(device)
    loss_sl_gauss_features = torch.zeros(1).to(device)
    loss_dsl_feat_lst = None
    loss_sl_gauss_feat_lst = None

    for step, (x) in enumerate(loader):
        step += start_step
        global_steps = epoch * steps_per_epoch + step
        x = x.to(device)  # images

        ##### stage 0: train E + G + Q
        opt_g.zero_grad()
        x_recon, loss_quant, logits_fake, _, enc_feats, dec_feats = model(x, stage=0)
        loss_l1 = (x - x_recon).abs().mean()
        loss_perceptual = lpips(x, x_recon).mean()

        loss_recon = loss_l1 + args.perceptual_weight * loss_perceptual
        loss_g = loss_recon + args.codebook_weight * loss_quant

        if epoch < args.disc_start_epochs:
            loss_disc = torch.tensor(0.).to(device)
            weight_d = torch.tensor(0.).to(device)
        else:
            loss_disc = hinge_g_loss(logits_fake)
            weight_d = compute_adaptive_weight(model, loss_recon, loss_disc, accelerator)
            loss_g = loss_g + weight_d * args.disc_weight * loss_disc

        if epoch < args.ffl_start_epochs:
            loss_ffl = torch.tensor(0.).to(device)
            loss_dsl_features = torch.tensor(0.).to(device)
            loss_sl_gauss_features = torch.tensor(0.).to(device)
        else:
            if args.ffl_weight > 0:
                loss_ffl = recon_ffl_loss(ffl_func, x, x_recon)
                loss_g = loss_g + loss_ffl
            if args.DSL_weight_features > 0:
                loss_dsl_features, loss_dsl_feat_lst = recon_ffl_features_loss(dsl_feature_func, enc_feats, dec_feats, device)
                loss_g = loss_g + loss_dsl_features
            if args.SL_weight > 0:
                loss_sl_gauss_features, loss_sl_gauss_feat_lst = recon_sl_gaussian_features_loss(sl_feature_func, args.gaussian_kernel, args.gaussian_sigma, enc_feats, dec_feats, device)
                loss_g = loss_g + loss_sl_gauss_features
        
        accelerator.backward(loss_g)
        opt_g.step()

        ##### stage 1: train D
        if epoch < args.disc_start_epochs:
            loss_d = torch.tensor(0.).to(device)
        else:
            opt_d.zero_grad()
            logits_real, logits_fake = model(x, stage=1)
            loss_d = hinge_d_loss(logits_real, logits_fake)
            accelerator.backward(loss_d)
            opt_d.step()

        losses = torch.tensor([loss_g, loss_recon, loss_l1, loss_perceptual, loss_ffl, loss_dsl_features, loss_sl_gauss_features, loss_quant, loss_disc, loss_d])
        loss_g, loss_recon, loss_l1, loss_perceptual, loss_ffl, loss_dsl_features, loss_sl_gauss_features, loss_quant, loss_disc, loss_d = [v.item() for v in losses]
        
        if step % args.print_steps == 0 and accelerator.is_main_process:
            mem_used = torch.cuda.max_memory_reserved() // (1 << 20)
            accelerator.print(f"Epoch: {epoch}, Step: {step}, loss_g: {loss_g:.3f}, loss_recon: {loss_recon:.3f}, loss_l1: {loss_l1:.3f}, "
                f"loss_perceptual: {loss_perceptual:.3f}, loss_ffl: {loss_ffl:.3f}, loss_dsl_features: {loss_dsl_features:.3f}, "
                f"loss_sl_gauss_features: {loss_sl_gauss_features:.3f}, loss_quant: {loss_quant:.3f}, loss_disc: {loss_disc:.3f}, loss_d: {loss_d:.3f}, "
                f"weight_d: {weight_d:.3f}, MemUsed: {mem_used} MiB", flush=True)

            # log to tensorboard
            if hasattr(model.module.encoder, 'sigmas'):
                enc_sigmas = model.module.encoder.sigmas.clone()
                writer.add_scalar("train/enc_sigma_0", enc_sigmas[0], global_steps)
                writer.add_scalar("train/enc_sigma_1", enc_sigmas[1], global_steps)
                writer.add_scalar("train/enc_sigma_2", enc_sigmas[2], global_steps)
                writer.add_scalar("train/enc_sigma_3", enc_sigmas[3], global_steps)

                dec_sigmas = model.module.decoder.sigmas.clone()
                writer.add_scalar("train/dec_sigma_0", dec_sigmas[0], global_steps)
                writer.add_scalar("train/dec_sigma_1", dec_sigmas[1], global_steps)
                writer.add_scalar("train/dec_sigma_2", dec_sigmas[2], global_steps)
                writer.add_scalar("train/dec_sigma_3", dec_sigmas[3], global_steps)

            if hasattr(model.module, 'sigmas'):
                sigmas = model.module.sigmas
                writer.add_scalar("train/sigma_0", sigmas[0], global_steps)
                writer.add_scalar("train/sigma_1", sigmas[1], global_steps)
                writer.add_scalar("train/sigma_2", sigmas[2], global_steps)
                writer.add_scalar("train/sigma_3", sigmas[3], global_steps)


            writer.add_scalar("train/loss_g", loss_g, global_steps)
            writer.add_scalar("train/loss_recon", loss_recon, global_steps)
            writer.add_scalar("train/loss_l1", loss_l1, global_steps)
            writer.add_scalar("train/loss_perceptual", loss_perceptual, global_steps)
            writer.add_scalar("train/loss_ffl", loss_ffl, global_steps)
            writer.add_scalar("train/loss_dsl_features", loss_dsl_features, global_steps)
            writer.add_scalar("train/loss_sl_gauss_features", loss_sl_gauss_features, global_steps)

            if loss_dsl_feat_lst is not None:
                writer.add_scalar("train/loss_dsl_features_block1", loss_dsl_feat_lst[0].item(), global_steps)
                writer.add_scalar("train/loss_dsl_features_block2", loss_dsl_feat_lst[1].item(), global_steps)
                writer.add_scalar("train/loss_dsl_features_block3", loss_dsl_feat_lst[2].item(), global_steps)
                writer.add_scalar("train/loss_dsl_features_block4", loss_dsl_feat_lst[3].item(), global_steps)

            if loss_sl_gauss_feat_lst is not None:
                writer.add_scalar("train/loss_sl_gauss_feat_lst_block1", loss_sl_gauss_feat_lst[0].item(), global_steps)
                writer.add_scalar("train/loss_sl_gauss_feat_lst_block2", loss_sl_gauss_feat_lst[1].item(), global_steps)
                writer.add_scalar("train/loss_sl_gauss_feat_lst_block3", loss_sl_gauss_feat_lst[2].item(), global_steps)
                writer.add_scalar("train/loss_sl_gauss_feat_lst_block4", loss_sl_gauss_feat_lst[3].item(), global_steps)
            
            writer.add_scalar("train/loss_quant", loss_quant, global_steps)
            writer.add_scalar("train/loss_disc", loss_disc, global_steps)
            writer.add_scalar("train/loss_d", loss_d, global_steps)
            writer.add_scalar("train/weight_d", weight_d, global_steps)

        # log the images to tensorboard
        if step % args.img_steps == 0:
            log_recons_image("train/img-recon", x, x_recon, global_steps, writer)


@torch.no_grad()
def validate(loader, model, lpips, epoch, perceptual_weight, writer, accelerator):
    model.eval()
    device = accelerator.device
    metrics = {}
    total, loss_l1s, loss_perceptuals, loss_recons = torch.zeros(4).to(device)

    total_steps = 0
    for (x) in loader:
        x = x.to(device)
        x_recon, _, _, _, _, _ = model(x, stage=0)
        loss_l1 = (x - x_recon).abs().mean()
        loss_perceptual = lpips(x, x_recon).mean()
        loss_recon = loss_l1 + perceptual_weight * loss_perceptual

        loss_l1s += loss_l1 * x.shape[0]
        loss_perceptuals += loss_perceptual * x.shape[0]
        loss_recons += loss_recon * x.shape[0]
        total += x.shape[0]

        total_steps += 1

    loss_recon = loss_recons.item() / total.item()
    loss_l1 = loss_l1s.item() / total.item()
    loss_perceptual = loss_perceptuals.item() / total.item()

    metrics['loss_recon'] = loss_recon
    metrics['loss_l1'] = loss_l1
    metrics['loss_perceptual'] = loss_perceptual

    if accelerator.num_processes > 1:
        # Then we should sync the metrics
        metrics_order = sorted(metrics.keys())
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        for i, metric_name in enumerate(metrics_order):
            metrics_tensor[0, i] = metrics[metric_name]
        metrics_tensor = accelerator.gather(metrics_tensor)
        metrics_tensor = metrics_tensor.mean(dim=0)
        for i, metric_name in enumerate(metrics_order):
            metrics[metric_name] = metrics_tensor[i].item()

    x = x.to(device)

    if accelerator.is_main_process:
        writer.add_scalar("val/loss_recon", loss_recon, epoch)
        writer.add_scalar("val/loss_l1", loss_l1, epoch)
        writer.add_scalar("val/loss_perceptual", loss_perceptual, epoch)
        log_recons_image("val/img-recon", x, x_recon, epoch, writer)

    accelerator.print(f"=== Validate: epoch {epoch}, loss_recon {metrics['loss_recon']:.3f}, loss_l1 {metrics['loss_l1']:.3f}, loss_perceptual {metrics['loss_perceptual']:.3f}", flush=True)

    return metrics['loss_recon']


def main(args, save_path):
    torch.manual_seed(0)
    ########################################
    # initialize accelerator
    ########################################
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    ########################################
    # initialize tensorboard
    ########################################

    log_path = os.path.join(save_path, "runs")
    writer = SummaryWriter(log_path, comment="lr_{}_bs_{}_pw_{}_cw_{}_dw_{}".format(args.base_lr, args.batch_size, args.perceptual_weight, args.codebook_weight, args.disc_weight))

    # set learning rate
    num_gpus = int(os.environ['WORLD_SIZE'])
    lr = args.base_lr * args.batch_size * num_gpus
    accelerator.print("\nlr = base_lr {} * batch size {} * num_gpus {} = {}".format(args.base_lr, args.batch_size, num_gpus, lr))

    # for L2 quantizer code
    sync_code = False
    if accelerator.state.num_processes > 1:
        num_processes = accelerator.state.num_processes
        accelerator.print("\nThe number of gpus used = ", num_processes)
        sync_code = True

    ########################################
    # initialize FA-VAE model
    ########################################
    if args.downsample_factor==16:
        ch_mult = (1,1,2,2,4)
        attn_resolutions=[16]
    elif args.downsample_factor==4:
        ch_mult=(1,2,4)
        attn_resolutions=[]
    elif args.downsample_factor==8:
        ch_mult=(1,2,2,4)
        attn_resolutions=[32]

    # with FCM 
    if args.with_fcm:
        model = VQGANFCM(args.codebook_size, args.embed_dim, args.double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, use_cosine_sim=args.use_cosine_sim, codebook_dim=args.codebook_dim,
                    orthogonal_reg_weight=args.orthogonal_reg_weight, orthogonal_reg_max_codes=args.orthogonal_reg_max_codes, use_l2_quantizer=args.use_l2_quantizer, sync_codebook=sync_code, 
                    commitment_weight=args.codebook_weight, use_non_pair_conv=args.use_non_pair_conv, kernel_size=args.gaussian_kernel, dsl_init_sigma=args.dsl_init_sigma, device=accelerator.device, 
                    use_gauss_resblock=args.use_gauss_resblock, use_gauss_attn=args.use_gauss_attn, use_same_conv_gauss=args.use_same_conv_gauss, use_same_gauss_resblock=args.use_same_gauss_resblock, 
                    use_patch_discriminator=args.use_patch_discriminator, disc_n_layers=args.disc_n_layers, use_ffl_with_fcm=args.use_ffl_with_fcm, num_groups=args.num_groups
                    )
    
    ########################################
    # initialize dataloaders
    ########################################
    train_loader, val_loader = load_data(args)

    ########################################
    # initialize optimizers
    ########################################
    # optimize E + G + Q
    g_params = list(model.encoder.parameters())   \
             + list(model.decoder.parameters())   \
             + list(model.quantizer.parameters()) 

    if hasattr(model, 'sigmas'):
        opt_g = torch.optim.Adam(
                            [{"params": g_params},
                             {"params": model.sigmas,'lr':2.0e-7}], lr=lr, betas=(0.5, 0.9))
    else:
        opt_g = torch.optim.Adam(g_params, lr=lr, betas=(0.5, 0.9))

    # optimize D
    d_params = model.discriminator.parameters()
    opt_d = torch.optim.Adam(d_params, lr=lr, betas=(0.5, 0.9))

    # perceptual loss
    lpips = LPIPS().cuda().eval()

    # initialize the SL/DSL losses
    ffl_func = None 
    if args.ffl_weight > 0:
        ffl_func = FFL(loss_weight=args.ffl_weight, alpha=1.0)  # initialize nn.Module class
        accelerator.print("LOSS SETTING: using FFL on the images level.")

    dsl_feature_func = None 
    if args.DSL_weight_features > 0:
        dsl_feature_func = FFL(loss_weight=args.DSL_weight_features, alpha=1.0)  # initialize nn.Module class
        if args.gaussian_kernel is not None:
            accelerator.print("LOSS SETTING: using DSL on the FCM features level with Gaussian kernel {} and initial sigma {}.".format(args.gaussian_kernel, args.dsl_init_sigma))
        else:
            accelerator.print("LOSS SETTING: using FFL on the FCM features level.")

    sl_feature_func = None
    if args.SL_weight > 0:
        sl_feature_func = FFL(loss_weight=args.SL_weight, alpha=1.0)  # initialize nn.Module class
        if args.gaussian_sigma is not None:
            accelerator.print("LOSS SETTING: using SL on the FCM features level with Gaussian kernel {} and fixed sigma {}.".format(args.gaussian_kernel, args.gaussian_sigma))
        else:
            accelerator.print("LOSS SETTING: using DSL on the FCM features level with Gaussian kernel {} and initial sigma {}.".format(args.gaussian_kernel, args.dsl_init_sigma))

    # load checkpoint to resume training
    start_epoch, start_step, best_score = 0, 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        if 'loss_recon' in ckpt:
            best_score = ckpt['loss_recon']
        accelerator.print(f"loaded model from epoch {start_epoch}, step {start_step}, best score {best_score}")
    
    # prepare the model and dataloaders for accelerator
    model = accelerator.prepare(model)
    opt_d, opt_g, train_loader, val_loader = accelerator.prepare(
        opt_d, opt_g, train_loader, val_loader
    )
    
    accelerator.print("\nOn distributed training, train loader ", len(train_loader))
    ########################################
    # Training
    ########################################
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # train
        train(train_loader, model, lpips, opt_g, opt_d, epoch, start_step, writer, accelerator, args, ffl_func, dsl_feature_func, sl_feature_func)
        start_step = 0  

        # validate
        loss_recon = validate(val_loader, model, lpips, epoch, args.perceptual_weight, writer, accelerator)

        # save the model
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if epoch % args.save_every_epoch == 0:
                state = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "epoch": epoch + 1,
                    "step": 0,
                    'loss_recon': loss_recon
                }
                save_model(state, os.path.join(save_path, "latest.pt"))

            if loss_recon < best_score:
                best_score = loss_recon
                save_model(state, os.path.join(save_path, "best.pt"))
                accelerator.print(f"New Best loss_recon: {loss_recon:.3f}", flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()

    if writer:
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FA-VAE")
    parser.add_argument("--ds", type=str, help="path to save outputs (ckpt, tensorboard runs)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--downsample_factor", type=int, default=16, help="downsample factor for FA-VAE")
    parser.add_argument("--save_every_epoch", type=int, default=1, help="save the checkpoint at every %\ epochs")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="the lpips weight")
    parser.add_argument("--disc_weight", type=float, default=0.8, help="the discriminator weight")
    parser.add_argument("--codebook_weight", type=float, default=1.0, help="the codebook weight")
    parser.add_argument("--disc_start_epochs", type=int, default=1, help="the number of epochs to start training the discriminator")
    parser.add_argument("--ffl_start_epochs", type=int, default=0, help="the number of epochs to start adding FFL and SL/DSL")
    parser.add_argument("--codebook_size", type=int, default=16384, help="the number of codebook entries")
    parser.add_argument("--embed_dim", type=int, default=256, help="the dimension of codebook entries")
    parser.add_argument("--codebook_dim", type=int, default=None, help="for projection in VitVQGAN: codebook dim is the dimension to be projected")
    parser.add_argument("--resolution", type=int, default=256, help="image resolution")
    parser.add_argument("--epochs", type=int, default=800, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loading")
    parser.add_argument("--print_steps", type=int, default=10, help="log training steps to tensorboard frequency")
    parser.add_argument("--img_steps", type=int, default=100, help="log training images to tensorboard frequency")
    parser.add_argument("--base_lr", type=float, default=4.5e-6, help="the base learninig rate, the actual lr = base_lr * batch_size * num_gpus")
    parser.add_argument("--resume", action='store_true', help="whether to resume from a checkpoint, must also provide argument for resume_path")
    parser.add_argument("--resume_path", type=str, help="the resume checkpoint path")
    parser.add_argument("--train_file", type=str, help="the training file, each entry contains (file_path, caption)")
    parser.add_argument("--test_file", type=str, help="the validation file")
    parser.add_argument("--jlib_file", type=str, help="same as training file, but is using jlib, only for ms-coco")
    parser.add_argument("--val_jlib_file", type=str, help="val file")
    parser.add_argument("--double_z", action='store_true', help="whether to double the channels in the encoder last block, set to true by default in taming, false in our experiments")
    parser.add_argument("--use_cosine_sim", action='store_true', help="for l2 regularization loss in VitVQGAN")
    parser.add_argument("--use_l2_quantizer", action='store_true', help="whether to use the l2 regularization quantizer")
    parser.add_argument("--with_fcm", action='store_true', help="whether to use FCM in the decoder")
    parser.add_argument("--use_non_pair_conv", action='store_true', help="non pair-wise DSL sigma with convolutional FCM.")
    parser.add_argument("--use_same_conv_gauss", action='store_true', help="use pair-wise DSL with convolutional FCM")
    parser.add_argument("--use_same_gauss_resblock", action='store_true', help="use pair-wise DSL with residual FCM")
    parser.add_argument("--use_gauss_resblock", action='store_true', help="use non pair-wise DSL with residual FCM")
    parser.add_argument("--use_gauss_attn", action='store_true', help="use non pair-wise DSL with attentio FCM")
    parser.add_argument("--use_ffl_with_fcm", action='store_true', help="use convolutional FCM with FFL in the decoder")
    parser.add_argument("--orthogonal_reg_active_codes_only", action='store_true', help="for orthogonal regularization loss")
    parser.add_argument("--orthogonal_reg_weight", type=float, default=0.0, help="orthogonal regularization weight, the paper recommends 10")
    parser.add_argument("--orthogonal_reg_max_codes", type=int, default=None, help="this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage")
    parser.add_argument("--ffl_weight", type=float, default=0.0, help="FFL weight on the images level")
    parser.add_argument("--DSL_weight_features", type=float, default=0.0, help="DSL/FFL weight on the FCM features level when using/not using Gaussian kernels")
    parser.add_argument("--SL_weight", type=float, default=0.0, help="DSL weight on the FCM features")
    parser.add_argument("--gaussian_kernel", type=int, default=None, help="the kernel size (mu) in DSL")
    parser.add_argument("--gaussian_sigma", type=int, default=None, help="the deterministic sigma in SL")
    parser.add_argument("--dsl_init_sigma", type=float, default=None, help="the initial sigma value in DSL for all fcm layers")
    parser.add_argument("--use_patch_discriminator", action='store_true', help="use PatchGAN as discriminator")
    parser.add_argument("--disc_n_layers", type=int, default=100, help="the number of layers in the discriminator")
    parser.add_argument("--num_groups", type=int, default=None, help="number of groups used for decoder with fcm")

    args = parser.parse_args()

    # the path to save the results
    save_path = os.path.join("output/{}/".format(args.ds))
    os.makedirs(save_path, exist_ok=True)
    # save the arguments into a json file
    with open('{}/train_cfg.yaml'.format(save_path), 'w') as file:
        yaml.dump(args, file)
    
    main(args, save_path)
