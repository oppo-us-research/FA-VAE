"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

"""
Cross-attention Autoregressive Transformer (CAT)
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None
    
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t + eps)
    
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size) -> None:
        super().__init__()

        self.encoder = nn.TransformerEncoderLayer(
            n_embd,
            n_head,
            n_embd * 4,
            dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        mask = ~torch.tril(mask)  # top-left is False, up-right is True
        self.register_buffer("mask", mask)

    def forward(self, x):
        L = x.size(1)
        assert L <= self.mask.size(0)
        mask = self.mask[:L, :L]

        return self.encoder(x, mask)

class Block_CA(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size) -> None:
        super().__init__()

        self.decoder = nn.TransformerDecoderLayer(
            n_embd,
            n_head,
            n_embd * 4,
            dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        mask = ~torch.tril(mask)  # top-left is False, up-right is True
        self.register_buffer("mask", mask)

    def forward(self, x, y, text_mask):
        L = x.size(1)
        assert L <= self.mask.size(0)
        mask = self.mask[:L, :L]

        return self.decoder(x, y, tgt_mask=mask, memory_mask=text_mask)

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)

class GPT(nn.Module):

    def __init__(self, vocab_size, n_layer, n_embed, n_head, dim_head=64, image_encoded_dim=16, n_cond_embed=768, dropout=0.1, max_text_len=128, cond_drop_prob = 0.25):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.image_encoded_dim = image_encoded_dim
        self.axial_height_pos = nn.Parameter(torch.randn(self.image_encoded_dim, n_embed))
        self.axial_width_pos = nn.Parameter(torch.randn(self.image_encoded_dim, n_embed))
        self.cond_proj = nn.Linear(n_cond_embed, n_embed)
        self.drop = nn.Dropout(dropout)
        self.max_text_len = max_text_len

        # CrossAttention required [SOS] token initialization
        self.start_token = nn.Parameter(torch.randn(n_embed))

        # blocks = [Block(n_embed, n_head, dropout, block_size) for _ in range(n_layer)]
        self.init_norm = LayerNorm(n_embed)
        self.blocks = nn.ModuleList([])
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleList([
                Attention(n_embed, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, dim_head = dim_head, heads = n_head, dropout = dropout),
                Attention(n_embed, context_dim = n_cond_embed, dim_head = dim_head, heads = n_head, dropout = dropout),
                FeedForward(n_embed, mult = 4, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(n_embed)

        self.to_logits = nn.Linear(n_embed, vocab_size, bias = False)
        self.to_logits.weight = self.tok_emb.weight

        self.cond_drop_prob = cond_drop_prob


    def forward(self, image_token_ids, text_token_embeds, text_mask, cond_drop_prob = None):
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        image_token_emb = self.tok_emb(image_token_ids)

        # add axial positional embedding

        axial_pos_emb = rearrange(self.axial_width_pos, 'w d -> 1 w d') + rearrange(self.axial_height_pos, 'h d -> h 1 d')
        axial_pos_emb = rearrange(axial_pos_emb, 'h w d -> (h w) d')

        batch, seq_len, device = *image_token_emb.shape[:2], image_token_emb.device

        image_token_emb = image_token_emb + axial_pos_emb[:seq_len]

        # add start token

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch)
        image_token_emb = torch.cat((start_tokens, image_token_emb), dim = 1)

        # text

        # enforce max text len

        text_token_embeds, text_mask = map(lambda t: t[:, :self.max_text_len], (text_token_embeds, text_mask))

        # classifier free guidance conditional dropout

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend

        x = image_token_emb
        x = self.init_norm(x)

        for self_attn, cross_attn, ff in self.blocks:
            x = self_attn(x) + x
            x = cross_attn(x, context = text_token_embeds, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)

        return logits


    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    @torch.no_grad()
    @eval_decorator
    def sample(self, embed, text_mask, temperature=1.0, top_k=None, top_p=1.0):
        batch = embed.size(0)

        image_seq_len = self.image_encoded_dim ** 2

        image_tokens = torch.empty((batch, 0), device = embed.device, dtype = torch.long)

        for _ in range(image_seq_len):
            logits = self.forward_with_cond_scale(
                text_token_embeds = embed,
                text_mask = text_mask,
                image_token_ids = image_tokens
            )[:, -1]

            filtered_logits = self.top_k_top_p(logits, top_k, top_p)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            image_tokens = torch.cat((image_tokens, sampled), dim = -1)

        image_tokens = rearrange(image_tokens, 'b (h w) -> b h w', h = self.image_encoded_dim)

        return image_tokens


    @staticmethod
    def top_k_top_p(logits, top_k=None, top_p=1.0):
        if top_k is not None:
            assert 1 <= top_k <= logits.size(-1)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[..., [-1]]] = -torch.inf

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            mask = cum_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0

            # scatter sorted tensors to original indexing
            mask = mask.scatter(1, sorted_indices, mask)
            logits[mask] = -torch.inf

        return logits


def gpt2_mini(vocab_size, dim_head=64, image_encoded_dim=16, n_cond_embed=768, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=24,
        n_embed=1536,
        n_head=24,
        dim_head = dim_head,
        image_encoded_dim=image_encoded_dim,
        n_cond_embed=n_cond_embed,
        dropout=dropout,
    )

def gpt2_medium(vocab_size, dim_head=64, image_encoded_dim=16, n_cond_embed=768, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=24,
        n_embed=1536,
        n_head=16,
        dim_head = dim_head,
        image_encoded_dim=image_encoded_dim,
        n_cond_embed=n_cond_embed,
        dropout=dropout,
    )


def gpt2_large(vocab_size, block_size=256, n_cond_embed=512, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=36,
        n_embed=1280,
        n_head=32,
        n_cond_embed=n_cond_embed,
        dropout=dropout,
    )
