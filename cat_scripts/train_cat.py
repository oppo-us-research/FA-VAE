"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0 
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""


"""
This script is to train CAT
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import sys
sys.path.append(".")
sys.path.append("..")
from models.txt_cond_transformer import Net2NetTransformer as N2NTransformer
from datasets.general_dataloader_gpt import load_data
from accelerate import Accelerator
from datasets.statistic import mean, std
from utils import *
import yaml
import matplotlib.pyplot as plt
from textwrap import wrap


def log_recon_images(name, log, step, writer):
    inputs = log["inputs"]
    x_sample = log["sample"]
    N = inputs.shape[0]

    std1 = torch.tensor(std).view(1, -1, 1, 1).to(x_sample)
    mean1 = torch.tensor(mean).view(1, -1, 1, 1).to(x_sample)
    inputs = (inputs * std1 + mean1)
    x_sample = (x_sample * std1 + mean1)

    inputs = inputs.permute(0, 2, 3, 1).clamp(0, 1)
    x_sample = x_sample.permute(0, 2, 3, 1).clamp(0, 1)

    fig = plt.figure(figsize=(40, 20))
    for i in range(inputs.shape[0]):
        temp = torch.cat([inputs[i], x_sample[i]], dim=0)
        temp = make_grid(temp, nrow=1).cpu()
        a = fig.add_subplot(1, N, i+1)
        plt.imshow(temp)
        a.axis("off")
        a.set_title("{}".format("\n".join(wrap(log['captions'][i], 30))), fontsize=10)

    writer.add_figure(name, fig, step)


def train(dataloader, gpt, optimizer, scheduler, scaler, epoch, start_step, writer, txt_cond, img_cond, txt_tok_cond, img_tok_cond, both_cond, accelerator):
    gpt.train()
    steps_per_epoch = len(dataloader) + start_step
    device = accelerator.device

    for step, (x, clip_x, txt) in enumerate(dataloader):
        gpt.train()
        
        step += start_step
        lr = scheduler.step(epoch + step / steps_per_epoch)
        batch = {}
        batch['image'] = x
        batch['clip_image'] = clip_x
        batch['caption'] = txt

        optimizer.zero_grad()
        x, c = gpt.get_xc(batch)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss_gpt = gpt(x, c)          # logits.shape = (1, 332,16384)

        accelerator.backward(loss_gpt)
        scaler.step(optimizer)
        scaler.update()

        total_steps = epoch * steps_per_epoch + step
        if step % args.print_steps == 0 and accelerator.is_main_process:
            mem_used = torch.cuda.max_memory_reserved() // (1 << 20)
            accelerator.print(f"Epoch: {epoch}, Step: {step}, loss_gpt: {loss_gpt:.3f}, lr: {lr:.7f}, MemUsed: {mem_used} MiB")

            writer.add_scalar("train/loss_gpt", loss_gpt, total_steps)
            writer.add_scalar("train/lr", lr, total_steps)

        if total_steps % args.img_steps == 0 and accelerator.is_main_process:
            gpt.eval()
            
            log = gpt.log_images(batch, top_k=args.top_k, top_p=args.top_p)
            log['captions'] = txt
            log_recon_images("train/from-cond", log, total_steps, writer)

    return loss_gpt


@torch.no_grad()
def validate(dataloader, gpt, epoch, writer, split, accelerator):
    gpt.eval()
    device = accelerator.device
    gpt = accelerator.unwrap_model(gpt).to(device)
    total, val_loss = torch.zeros(2).to(device)

    metrics = {}
    steps = 0
    for step, (x, clip_x, txt) in enumerate(dataloader):
        batch = {}
        batch['image'] = x
        batch['clip_image'] = clip_x
        batch['caption'] = txt

        x, c = gpt.get_xc(batch)
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss_gpt = gpt(x, c)  

        val_loss += loss_gpt * x.shape[0]
        total += x.shape[0]

        steps += 1

    val_loss = val_loss.item() / total.item()

    metrics['val_gpt_loss'] = val_loss
    if accelerator.num_processes > 1:
        metrics_order = sorted(metrics.keys())
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        for i, metric_name in enumerate(metrics_order):
            metrics_tensor[0, i] = metrics[metric_name]
        metrics_tensor = accelerator.gather(metrics_tensor)
        metrics_tensor = metrics_tensor.mean(dim=0)
        for i, metric_name in enumerate(metrics_order):
            metrics[metric_name] = metrics_tensor[i].item()

    if accelerator.is_main_process:
        accelerator.print(f"=== Validate: epoch {epoch}, val_loss {val_loss:.3f}")

        # decode the last batch
        log = gpt.log_images(batch, top_k=args.top_k, top_p=args.top_p)
        log['captions'] = txt
        writer.add_scalar("val/{}_val_loss".format(split), val_loss, epoch)
        log_recon_images("val/from-cond", log, epoch, writer)

    return metrics['val_gpt_loss']


def main(args, save_path):
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    log_path = os.path.join(save_path, "runs")

    torch.manual_seed(0)

    writer = SummaryWriter(log_path)

    sync_code = False
    num_processes = accelerator.state.num_processes
    if num_processes > 1:
        print("\nThe number of gpus used = ", num_processes)
        sync_code = True

    ##################################
    # initialize FA-VAE & CAT models together
    ##################################
    lr = args.base_lr * args.batch_size * num_processes

    print("Setting learning rate to {:.2e} = {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            lr, num_processes, args.batch_size,  args.base_lr))
    gpt = N2NTransformer(args, sync_code=sync_code, device=device, learning_rate=lr, accelerator=accelerator)

    train_loader, val_loader = load_data(args)

    ##################################
    # scaler & optimizer
    ##################################
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    optimizer = gpt.optimizer
    scheduler = CosineLRWarmUp(optimizer, args.warmup_epochs, args.epochs, lr, args.min_lr, enabled=args.enabled_warmup)

    best_score = torch.inf
    best_train_score = torch.inf
    start_epoch, start_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume_path, map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_score = ckpt["best_score"]
    else:
        print(f"gpt ckpt not found, start training from scratch")

    scaler, scheduler, train_loader, val_loader = accelerator.prepare(
        scaler, scheduler, train_loader, val_loader
    )

    # train, validate
    for epoch in range(start_epoch, args.epochs):

        tr_loss_gpt = train(train_loader, gpt, optimizer, scheduler, scaler, epoch, start_step, writer, args.txt_cond, args.img_cond, args.txt_tok_cond, args.img_tok_cond, args.both_cond, accelerator)
        val_loss = validate(val_loader, gpt, epoch, writer, "text-condition", accelerator)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            
            if epoch % args.save_every_epoch == 0:
                latest_path = os.path.join(save_path, "latest.pt")
                state = {
                    "transformer_model": accelerator.unwrap_model(gpt.transformer).state_dict(),
                    "epoch": epoch + 1,
                    "best_score": min(best_score, val_loss),
                    "step": 0,
                }
                save_model(state, latest_path)

            if val_loss < best_score:
                best_score = val_loss
                save_model(state, os.path.join(save_path, "best.pt"))
                accelerator.print(f"New Best loss_gpt: {best_score:.3f}", flush=True)

            if tr_loss_gpt < best_train_score:
                best_train_score = tr_loss_gpt
                save_model(state, os.path.join(save_path, "best_train.pt"))
                accelerator.print(f"New Best loss_train_gpt: {best_train_score:.3f}", flush=True)

        accelerator.wait_for_everyone()

    accelerator.end_training()

    if writer:
        writer.close()


if __name__ == "__main__":

    ###########################################
    # CONFIG
    ###########################################
    parser = argparse.ArgumentParser(description="Train CAT")
    parser.add_argument("--ds", type=str, help="path to save outputs (ckpt, tensorboard runs)")
    parser.add_argument("--gpt_name", type=str, default="gpt2_medium", help="GPT backbone to be used")
    parser.add_argument("--clip", type=str, default=None, help="clip model, Vit-B32, Vit-L14")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--min_lr", type=float, default=0., help="minimum learning rate for the scheduler")
    parser.add_argument("--base_lr", type=float, default=2e-6, help="the base learninig rate, the actual lr = base_lr * batch_size * num_gpus")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--save_every_epoch", type=int, default=1, help="save the checkpoint at every %\ epochs")
    parser.add_argument("--favae_ckpt", type=str, help="VQGAN pretrained model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="the number of codebook entries")
    parser.add_argument("--embed_dim", type=int, default=256, help="the dimension of codebook entries")
    parser.add_argument("--double_z", action='store_true', help="whether to double the channels in the encoder last block, set to true by default in taming, false in our experiments")
    parser.add_argument("--n_layer", type=int, default=None, help="number of layers in the GPT model")
    parser.add_argument("--n_head", type=int, default=None, help="number of heads in the GPT model")
    parser.add_argument("--n_embd", type=int, default=None, help="embedding size of GPT model")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loading")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="number of warmup epochs for the scheduler")
    parser.add_argument("--resolution", type=int, default=256, help="image resolution")
    parser.add_argument("--top_k", type=int, default=500, help="top k for image generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="top p for image generation")
    parser.add_argument("--normalize_clip", action='store_true', help="whether to normalize the clip embeddings")
    parser.add_argument("--use_amp", action='store_true', help="use automatic mixed precision")
    parser.add_argument("--enabled_warmup", action='store_true', help="enable warmup for scheduler")
    parser.add_argument("--print_steps", type=int, default=10, help="log training steps to tensorboard frequency")
    parser.add_argument("--img_steps", type=int, default=100, help="log training images to tensorboard frequency")
    parser.add_argument("--img_cond", action='store_true', help="use CLIP image embeddings as condition")
    parser.add_argument("--txt_cond", action='store_true', help="use CLIP text embeddings as condition")
    parser.add_argument("--txt_tok_cond", action='store_true', help="use CLIP text token embeddings as condition")
    parser.add_argument("--img_tok_cond", action='store_true', help="use CLIP image token embeddings as condition")
    parser.add_argument("--both_cond", action='store_true', help="use both CLIP text and image embeddings as condition")
    parser.add_argument("--unconditional", action='store_true', help="unconditional GPT training")
    parser.add_argument("--cls_cond", action='store_true', help="class conditioned GPT training")
    parser.add_argument("--n_classes", type=int, default=None, help="number of classed for class condition GPT")
    parser.add_argument("--resume", action='store_true', help="whether to resume from a checkpoint, must also provide argument for resume_path")
    parser.add_argument("--resume_path", type=str, default=None, help="the resume checkpoint path")
    parser.add_argument("--train_file", type=str, help="the training file, each entry contains (file_path, caption)")
    parser.add_argument("--val_file", type=str, help="the validation file")
    parser.add_argument("--use_cosine_sim", action='store_true', help="for l2 regularization loss in VitVQGAN")
    parser.add_argument("--use_l2_quantizer", action='store_true', help="whether to use the quantizer code from Lucidrains: https://github.com/lucidrains/vector-quantize-pytorch")
    parser.add_argument("--codebook_dim", type=int, default=256, help="for projection in VitVQGAN: codebook dim is the dimension to be projected")
    parser.add_argument("--orthogonal_reg_active_codes_only", action='store_true', help="for orthogonal regularization loss")
    parser.add_argument("--orthogonal_reg_weight", type=float, default=0.0, help="orthogonal regularization weight, the paper recommends 10")
    parser.add_argument("--orthogonal_reg_max_codes", type=int, default=None, help="this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage")
    parser.add_argument("--use_same_conv_gauss", action='store_true', help="use pair-wise DSL with convolutional FCM")
    parser.add_argument("--use_same_gauss_resblock", action='store_true', help="use pair-wise DSL with residual FCM")
    parser.add_argument("--use_gauss_resblock", action='store_true', help="use non pair-wise DSL with residual FCM")
    parser.add_argument("--use_gauss_attn", action='store_true', help="use non pair-wise DSL with attentio FCM")
    parser.add_argument("--use_patch_discriminator", action='store_true', help="use PatchGAN as discriminator, used by default in taming")
    parser.add_argument("--gaussian_kernel", type=int, default=None, help="the kernel size (mu) in DSL")
    parser.add_argument("--gaussian_sigma", type=int, default=None, help="the initial sigma (sigma) in DSL")
    parser.add_argument("--use_single_fcm_block", action='store_true', help="add only one FCM block to the decoder, must supply argument for the fcm_block_idx")
    parser.add_argument("--fcm_block_idx", type=int, default=None, help="the block to add the FCM, idx from {0,1,2,3}")
    parser.add_argument("--n_cond_embed", type=int, default=None, help="condition embedding size")
    parser.add_argument("--disc_n_layers", type=int, default=3, help="the number of layers in the discriminator")
    parser.add_argument("--sub_val_file", type=str, help="only contain caption, for both condition training")
    parser.add_argument("--downsample_factor", type=int, default=16, help="downsample factor for FA-VAE")
    parser.add_argument("--num_groups", type=int, default=None, help="number of groups used for decoder with fcm")
    parser.add_argument("--dsl_init_sigma", type=float, default=3.0, help="the initial sigma value in DSL for all fcm layers, not used in CAT training")
    args = parser.parse_args()

    save_path = Path(f"output/our_gpt_CA_ddp/{args.ds}")
    save_path.mkdir(exist_ok=True, parents=True)
    with open('{}/train_cfg.yaml'.format(save_path), 'w') as file:
        yaml.dump(args, file)

    main(args, save_path)