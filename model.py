from __future__ import annotations

import math
import pickle
import struct
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from tokenizer import SmilesTokenizer


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


@dataclass
class ContextArgs:
    context_keys: List[str] = field(default_factory=list)
    context_dims: List[int] = field(default_factory=list)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.cache_hash = None

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = (
                scores + self.mask[:, :, :seqlen, :seqlen]
            )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    def forward_with_kvcache(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        cache_id: int = 1,
    ):
        bsz, seqlen, _ = x.shape

        original_x = x
        use_cache = self.cache_hash == cache_id
        if use_cache:
            x = x[:, -1, :].unsqueeze(1)  # only need the last new token
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if use_cache:
            # comp_xq, comp_xk, comp_xv = self.wq(original_x), self.wk(original_x), self.wv(original_x)
            # comp_xq = comp_xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            # comp_xk = comp_xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            # comp_xv = comp_xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            # # RoPE relative positional embeddings
            # comp_xq, comp_xk = apply_rotary_emb(comp_xq, comp_xk, freqs_cos, freqs_sin)

            self.k_cache = torch.concat([self.k_cache, xk.clone()], dim=1)
            self.v_cache = torch.concat([self.v_cache, xv.clone()], dim=1)
            # print("Before positional xk:", torch.all(self.k_cache == self.wk(original_x)))
            # print("Before positional xv:", torch.all(self.v_cache == self.wv(original_x)))

            seqlen = self.k_cache.size(1)
            xk = self.k_cache
            xv = self.v_cache
            self.cache_hash = cache_id
            xq = xq.view(bsz, 1, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            # RoPE relative positional embeddings
            # xq, xk = apply_rotary_emb(xq, xk[:,-1,:,:].unsqueeze(1), freqs_cos[-1,:].unsqueeze(0), freqs_sin[-1,:].unsqueeze(0))
            # reshape xq and xk to match the complex representation
            xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
            xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

            # reshape freqs_cos and freqs_sin for broadcasting
            q_freq_cos = freqs_cos[-1, :].unsqueeze(0)
            q_freq_sin = freqs_sin[-1, :].unsqueeze(0)
            freqs_cos_q = reshape_for_broadcast(q_freq_cos, xq_r)
            freqs_sin_q = reshape_for_broadcast(q_freq_sin, xq_r)

            freqs_cos_k = reshape_for_broadcast(freqs_cos, xk_r)
            freqs_sin_k = reshape_for_broadcast(freqs_sin, xk_r)

            # apply rotation using real numbers
            xq_out_r = xq_r * freqs_cos_q - xq_i * freqs_sin_q
            xq_out_i = xq_r * freqs_sin_q + xq_i * freqs_cos_q
            xk_out_r = xk_r * freqs_cos_k - xk_i * freqs_sin_k
            xk_out_i = xk_r * freqs_sin_k + xk_i * freqs_cos_k

            # flatten last two dimensions
            xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
            xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

            xq, xk = xq_out.type_as(xq), xk_out.type_as(xk)
            # print(f"Seq len {xk.shape[1]} xq:", torch.allclose(xq , comp_xq[:,-1,:].unsqueeze(1), atol=1e-7), torch.mean(xq - comp_xq[:,-1,:].unsqueeze(1)))
            # print(f"Seq len {xk.shape[1]} xk:",  torch.allclose(xk ,comp_xk, atol=1e-7), torch.mean(xk - comp_xk))
            # print(f"Seq len {xk.shape[1]} xv:",  torch.allclose(xv , comp_xv, atol=1e-7), torch.mean(xv - comp_xv))
            # print("-"*10)
            # self.old_x = original_x
        else:
            self.k_cache = xk
            self.v_cache = xv
            self.old_x = x

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            self.cache_hash = cache_id

            # RoPE relative positional embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                # NOTE: VERY IMPORTANT to set is_causal=False, OTHERWISE the KV-Caching just breaks
                is_causal=False,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = (
                scores + self.mask[:, :, :seqlen, :seqlen]
            )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        # if use_cache:
        #     # original_x[:,-1,:] = output.transpose(1, 2).contiguous().view(bsz,-1)
        #     # output = original_x
        #     output = torch.concat( [self.out_cache, output.transpose(1, 2).view(bsz,1,-1)], dim=1).contiguous()
        #     self.out_cache = output
        # else:
        #     output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        #     self.out_cache = output

        # NOTE: only work when fed in one token at a time (e.g. seq = 1)
        output = output.transpose(1, 2).contiguous().view(bsz, x.size(1), -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

    def forward_with_kvcache(self, x, freqs_cos, freqs_sin, cache_id=1):
        h = x + self.attention.forward_with_kvcache(
            self.attention_norm(x), freqs_cos, freqs_sin, cache_id=cache_id
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs, context_params: ContextArgs):
        super().__init__()
        self.params = params
        self.context_params = context_params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.frag_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.frag_type_embedding = nn.Embedding(1, params.dim)

        self.context_lookup = {k: i for i, k in enumerate(context_params.context_keys)}
        self.conditions_type_embeddings = nn.Embedding(
            len(context_params.context_keys), params.dim
        )
        self.conditions_embeddings_lookup = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(dim, params.dim, bias=True),
                )
                for k, dim in zip(
                    context_params.context_keys, context_params.context_dims
                )
            }
        )

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = (
            self.output.weight
        )  # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers)
                )

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        fragment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        device = tokens.device

        h = self._add_context_to_seq(tokens, context, fragment, bsz, device)

        context_seq_len = h.shape[1] - seqlen

        bsz, seqlen, _ = h.shape

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        h = h[:, context_seq_len:]
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            tmp_last_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # Ignore Pad Tokens
            )

            # NOTE: This essentially does nothing for the computation,
            # because we are multiplying the weights by zero.
            # This *needs* to be done, so that we can train with DDP
            # As due to the random training process some of the weights are not used in the forward pass
            # That is unacceptable for the for the c10 backend and the training errors out.
            # Maybe there is a better fix in the future, see:
            # https://github.com/pytorch/pytorch/issues/43259
            ddp_fix = sum(p.sum() for p in self.parameters())
            zero_sum = ddp_fix * 0.0

            self.last_loss = tmp_last_loss + zero_sum
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(
                h[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def forward_with_kvcache(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        fragment: Optional[torch.Tensor] = None,
        cache_id: int = 1,
        pos_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        device = tokens.device

        h = self._add_context_to_seq(tokens, context, fragment, bsz, device)

        context_seq_len = h.shape[1] - seqlen

        bsz, seqlen, _ = h.shape
        if pos_seq_len is None:
            pos_seq_len = seqlen
        else:
            pos_seq_len = max(seqlen, pos_seq_len + context_seq_len)

        freqs_cos = self.freqs_cos[:pos_seq_len]
        freqs_sin = self.freqs_sin[:pos_seq_len]

        for layer in self.layers:
            h = layer.forward_with_kvcache(h, freqs_cos, freqs_sin, cache_id=cache_id)
        h = self.norm(h)

        h = h[:, context_seq_len:]
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            tmp_last_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # Ignore Pad Tokens
            )

            # NOTE: This essentially does nothing for the computation,
            # because we are multiplying the weights by zero.
            # This *needs* to be done, so that we can train with DDP
            # As due to the random training process some of the weights are not used in the forward pass
            # That is unacceptable for the for the c10 backend and the training errors out.
            # Maybe there is a better fix in the future, see:
            # https://github.com/pytorch/pytorch/issues/43259
            ddp_fix = sum(p.sum() for p in self.parameters())
            zero_sum = ddp_fix * 0.0

            self.last_loss = tmp_last_loss + zero_sum
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(
                h[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def _add_context_to_seq(self, tokens, context, fragment, bsz, device):
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        if fragment is not None:
            fragment_type_enc = torch.zeros_like(
                fragment, dtype=torch.long, device=device
            )

            h = torch.concat(
                (
                    self.tok_embeddings(fragment)
                    + self.frag_embeddings(fragment)
                    + self.frag_type_embedding(fragment_type_enc),
                    h,
                ),
                dim=1,
            )

        if context is not None and len(context) != 0:
            # context is a dictionary with key : context_tensor of shape (batch_size, context_dim)
            type_ids = []
            context_vals = []

            for emb_key, context_val in context.items():
                emb_context_val = self.conditions_embeddings_lookup[emb_key](
                    context_val.unsqueeze(1).to(device)
                ).unsqueeze(1)

                context_vals.append(emb_context_val)
                type_ids_tensor = torch.tensor(
                    [self.context_lookup[emb_key]], device=device, dtype=torch.long
                )
                type_ids.append(type_ids_tensor)

            context_types = (
                torch.concat(type_ids, dim=0).reshape(-1, 1).expand(-1, bsz).T
            )
            # shape(len(context),batch_size, emb_size)
            context_types = self.conditions_type_embeddings(context_types)

            context_vals = torch.concat(context_vals, dim=1).to(device)

            # SHAPE
            h = torch.concat([context_vals + context_types, h], dim=1)
        return h

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(
        self,
        tokenizer: SmilesTokenizer,
        context: Union[torch.Tensor, None] = None,
        fragments: Union[torch.Tensor, None] = None,
        max_length: int = 50,
        num_gen: int = 200,
        start_smiles: Union[str, None] = None,
        temperature: float = 1.0,
        top_k: Union[int, None] = None,
        device: torch.device = torch.device("cpu"),
        cache_kv: bool = False,
    ) -> List[str]:
        batch_size = num_gen
        if start_smiles is not None:
            tokenized_start_selfie = tokenizer.encode(start_smiles)[
                :-1
            ]  # remove <eos> token
            tokenized_start_selfie = torch.tensor(
                tokenized_start_selfie, device=device, dtype=torch.long
            ).view(-1, 1)
            tokenized_start_selfie = tokenized_start_selfie.repeat(1, batch_size)

            outputs = tokenized_start_selfie.T
        else:
            outputs = (
                torch.LongTensor([[tokenizer.cls_token_id] * batch_size]).to(device)
            ).T  # batch_size
        self.eval()

        start_len = outputs.shape[1]
        has_end_idx = np.array([0] * batch_size)
        cache_id = np.random.randint(0, int(1e10), 1).item()
        with torch.no_grad():
            with tqdm(total=max_length, desc="Generation") as pbar:
                for i in range(start_len, max_length):
                    # trg_tensor = #torch.LongTensor(outputs).to(model.device)
                    if not cache_kv:
                        logits = self(outputs, context=context, fragment=fragments)
                    else:
                        # logits_ = self(outputs, context=context, fragment=fragments)
                        if i == start_len:
                            # When starting pass the whole input, so that "start_smiles" works, then only the newly generated token, because of the cache
                            func_input = outputs
                        else:
                            func_input = outputs[:, -1].unsqueeze(-1)
                        logits = self.forward_with_kvcache(
                            func_input,
                            context=context,
                            fragment=fragments,
                            cache_id=cache_id,
                            pos_seq_len=outputs.size(-1),
                        )

                        # raise NotImplementedError("Currently not working / right implemented")
                        # logits = self.forward_with_kvcache(outputs, context=context, fragment=fragments,cache_id = cache_id)

                    logits = logits[:, -1, :]  # crop to just the final time step
                    if temperature == 0.0:
                        # "sample" the single most likely index
                        _, logits = torch.topk(logits, k=1, dim=-1)
                    else:
                        # pluck the logits at the final step and scale by desired temperature
                        logits = logits / temperature
                        # optionally crop the logits to only the top k options
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float("Inf")

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    ended_sentences = idx_next == tokenizer.sep_token_id
                    if torch.count_nonzero(ended_sentences) != 0:
                        indicies = torch.nonzero(ended_sentences)
                        indicies = indicies.cpu().numpy()
                        for end_idx in indicies[:, 0]:
                            if has_end_idx[end_idx] == 0:
                                has_end_idx[end_idx] = i

                        # print(has_end_idx)

                    if all([idx != 0 for idx in has_end_idx]):
                        break

                    # outputs.append(best_guesses)
                    # outputs = torch.row_stack((outputs, idx_next))
                    outputs = torch.cat((outputs, idx_next), dim=1)
                    pbar.update(1)

        out_selfies = []
        for output, end_idx in zip(outputs.cpu().numpy(), has_end_idx):
            # Incase of limiting the max_len
            if end_idx == 0:
                selfie = [tokenizer._convert_id_to_token(idx) for idx in output[:]]
            else:
                selfie = [
                    tokenizer._convert_id_to_token(idx) for idx in output[:end_idx]
                ]
            selfie = "".join(selfie[1:])
            out_selfies.append(selfie)

        # for indicies in outputs:
        #     translated_sentence = [tokenizer.idx_to_tokens[idx]  for idx in outputs]
        # remove start token
        return out_selfies

    @staticmethod
    def load(path, device: torch.device = torch.device("cpu")) -> Transformer:
        data = torch.load(path, map_location=device)

        newinstace = Transformer(data["model_params"], data["context_params"])
        newinstace.load_state_dict(data["state_dict"])
        return newinstace.to(device)

    def save(self, filepath):
        torch.save(
            {
                "state_dict": self.state_dict(),
                **dict(model_params=self.params, context_params=self.context_params),
            },
            filepath,
        )

    def getNumberTrainableParams(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def getNumberParams(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = Transformer(
        ModelArgs(dim=128, n_layers=8, n_heads=8, vocab_size=512, max_seq_len=1024),
        context_params=ContextArgs(
            context_keys=["logp", "sascore", "mol_weight"], context_dims=[1, 1, 1]
        ),
    )
    seq = torch.ones((128, 50), dtype=torch.long)
    frag = torch.ones((128, 10), dtype=torch.long)
    context = {
        "logp": torch.ones((128,), dtype=torch.float32),
        # "sascore": torch.ones((128,), dtype=torch.float32),
        "mol_weight": torch.ones((128,), dtype=torch.float32),
    }

    print(m.forward(seq, targets=seq, context=context, fragment=frag))
