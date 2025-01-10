
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        mem = self.norm2(mem)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

class PromptGeneration(nn.Module):
    def __init__(
        self,
        d_model,output_dim
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x



class ProtoFusion(nn.Module):
    def __init__(
        self,
        d_model,output_dim
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class ProtoFusion(nn.Module):
    def __init__(
        self,
        d_model,output_dim
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x