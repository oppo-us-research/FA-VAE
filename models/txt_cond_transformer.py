"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

"""
This is for text conditional GPT, code adopted from taming cond_transformer.py.
modified to suit text conditional generation.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append(".")
sys.path.append("..")
from models.vqgan_fcm import VQGANFCM
from utils import CLIPWrapper
from CLIP.clip.clip_custom import tokenize

from models.gpt_ca import __dict__ as gpts

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(torch.nn.Module):
    def __init__(self,
                args,
                sync_code=False,
                device=None,
                permuter_config=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                pkeep=1.0,
                sos_token=0,
                learning_rate=None,
                accelerator=None
                ):
        super().__init__()
        self.args = args
        self.be_unconditional = args.unconditional
        self.txt_tok_cond = args.txt_tok_cond
        self.cls_cond = args.cls_cond
        self.sync_code = sync_code
        self.device = device
        self.sos_token = sos_token
        self.learning_rate = learning_rate
        self.accelerator = accelerator

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(args)
        self.init_cond_stage_from_ckpt(args)
        self.permuter = torch.nn.Identity()
        self.init_gpt(args)
        self.optimizer = self.configure_optimizers()
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.transformer = self.accelerator.prepare(self.transformer)

    def init_first_stage_from_ckpt(self, args):
        if args.downsample_factor==16:
            ch_mult = (1,1,2,2,4)
            attn_resolutions=[16]
        elif args.downsample_factor==4:
            ch_mult=(1,2,4)
            attn_resolutions=[]
        elif args.downsample_factor==8:
            ch_mult=(1,2,2,4)
            attn_resolutions=[32]
        favae = VQGANFCM(args.codebook_size, args.embed_dim, args.double_z, ch_mult=ch_mult, attn_resolutions=attn_resolutions, use_cosine_sim=args.use_cosine_sim, codebook_dim=args.codebook_dim,
                    orthogonal_reg_weight=args.orthogonal_reg_weight, orthogonal_reg_max_codes=args.orthogonal_reg_max_codes, use_l2_quantizer=args.use_l2_quantizer, sync_codebook=self.sync_code, 
                    kernel_size=args.gaussian_kernel, device=self.device, use_gauss_resblock=args.use_gauss_resblock, use_gauss_attn=args.use_gauss_attn, 
                    use_same_conv_gauss=args.use_same_conv_gauss, use_same_gauss_resblock=args.use_same_gauss_resblock, dsl_init_sigma=args.dsl_init_sigma,
                    use_patch_discriminator=args.use_patch_discriminator, disc_n_layers=args.disc_n_layers, num_groups=args.num_groups, inference=True
                    ).eval().requires_grad_(False).to(self.device)
        state = torch.load(args.favae_ckpt, map_location='cpu')
        favae.load_state_dict(state['model'])
        print(f"Loaded favae model from {args.favae_ckpt}, epoch {state['epoch']}")

        self.first_stage_model = favae

    def init_cond_stage_from_ckpt(self, args):
        if self.txt_tok_cond:
            from CLIP.clip import clip_custom
            if args.clip == 'vit-b-32':
                clip_model, _ = clip_custom.load("ViT-B/32", device=self.device)
            elif args.clip == 'vit-l-14':
                clip_model, _ = clip_custom.load("ViT-L/14", device=self.device)

            clip_model = CLIPWrapper(clip_model, normalize=args.normalize_clip)
            self.cond_stage_key = 'caption'
            self.cond_stage_model = clip_model
        else:
            raise NotImplementedError

    def init_gpt(self, args):
        gpt = gpts[args.gpt_name](vocab_size=args.codebook_size, dropout=args.dropout, n_cond_embed=args.n_cond_embed)
        if args.resume_path is not None:
            state = torch.load(args.resume_path,  map_location='cpu')
            gpt.load_state_dict(state['transformer_model'])
            print(f"Loaded GPT model from {args.resume_path}, epoch {state['epoch']}, best score {state['best_score']}")

        self.transformer = gpt

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        text_embeddings, text_mask = self.encode_to_c(c)

        L = z_indices.size(1)
        input_tokens = z_indices[:, :(L - 1)].contiguous()  # [B, L]

        # make the prediction
        logits = self.transformer(input_tokens, text_embeddings, text_mask)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        loss_gpt = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))

        return loss_gpt

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out


    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, indices, _ = self.first_stage_model.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices


    @torch.no_grad()
    def get_cond_emb(self, texts):
        texts = tokenize(texts).cuda()
        text_mask = torch.where(texts>0,1,0)
        text_mask = text_mask.to(torch.bool)
        txt_tok_emb, _ = self.cond_stage_model.encode_text(texts)
        txt_tok_emb = txt_tok_emb.to(torch.float32)

        return txt_tok_emb, text_mask

    @torch.no_grad()
    def encode_to_c(self, c):
        text_embeddings, text_mask = self.get_cond_emb(c)
        text_embeddings = text_embeddings.to(self.device)
        text_mask = text_mask.to(self.device)

        return text_embeddings, text_mask

    @torch.no_grad()
    def decode_to_img(self, index, zshape):    # index.shape = (4, 256)   zshape = (4, 256, 16,16)
        index = self.permuter(index)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])

        quant_z = self.first_stage_model.quantizer.get_codebook_entry(
            index, shape=bhwc)
        x, _ = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, top_p=None, callback=None, lr_interface=False, **kwargs):
        log = dict()
        base_transformer = self.accelerator.unwrap_model(self.transformer)
        x, c = self.get_xc(batch)
        quant_z, z_indices = self.encode_to_z(x)
        text_embeddings, text_mask = self.encode_to_c(c)
        z_idx = base_transformer.sample(text_embeddings, text_mask, top_k=top_k, top_p=top_p)  # [B, 16*16]

        with torch.no_grad():
            z_idx = z_idx.view(-1, 16, 16)      # z_idx.shape = (25,16,16)
            x_sample = self.decode_to_img(z_idx, quant_z.shape)

        log['inputs'] = x
        log['sample'] = x_sample
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = batch[self.cond_stage_key]
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result


    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist = (torch.nn.LayerNorm, torch.nn.Embedding)
        decay_parameters = self.get_parameter_names(self.transformer, [torch.nn.LayerNorm, torch.nn.Embedding])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optim_groups= [
                    {
                        "params": [p for n, p in self.transformer.named_parameters() if n in decay_parameters],
                        "weight_decay": 0.01,
                    },
                    {
                        "params": [p for n, p in self.transformer.named_parameters() if n not in decay_parameters],
                        "weight_decay": 0.0,
                    },
                ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        return optimizer