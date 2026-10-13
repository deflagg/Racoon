from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RaccoonConfig:
    vocab_size: int = 8
    max_seq_len: int = 8
    d_model: int = 4
    n_heads: int = 1
    mlp_hidden: int = 8


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RaccoonConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate_attention: bool = False,
        return_attn: bool = False,
    ):
        bsz, seq_len, d_model = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        y = self.o_proj(y)

        if ablate_attention:
            y = torch.zeros_like(y)

        if return_attn:
            return y, attn
        return y


class RaccoonBlock(nn.Module):
    def __init__(self, cfg: RaccoonConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=True)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ablate_attention: bool = False,
        return_attn: bool = False,
    ):
        if return_attn:
            attn_out, attn = self.attn(
                self.ln1(x),
                ablate_attention=ablate_attention,
                return_attn=True,
            )
        else:
            attn_out = self.attn(
                self.ln1(x),
                ablate_attention=ablate_attention,
                return_attn=False,
            )
            attn = None

        x = x + attn_out
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

        if return_attn:
            return x, attn
        return x


class Raccoon(nn.Module):
    """Frozen 244-parameter one-block Transformer baseline."""

    def __init__(self, cfg: RaccoonConfig = RaccoonConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.block = RaccoonBlock(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        ablate_attention: bool = False,
        return_attn: bool = False,
    ):
        _, seq_len = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        pos = torch.arange(seq_len, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]

        if return_attn:
            x, attn = self.block(
                x,
                ablate_attention=ablate_attention,
                return_attn=True,
            )
        else:
            x = self.block(
                x,
                ablate_attention=ablate_attention,
                return_attn=False,
            )
            attn = None

        x = self.ln_f(x)

        # Weight tying: no additional language-model-head parameters.
        logits = x @ self.token_emb.weight.T

        if return_attn:
            return logits, attn
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Raccoon()
    n = count_parameters(model)
    print(model)
    print(f"trainable parameters: {n}")
    assert n == 244, f"expected 244 parameters, got {n}"
