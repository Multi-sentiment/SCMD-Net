"""
Context Transformer with Rotary Position Embedding (RoPE)
K-layer Transformer for modality-specific context enhancement.
"""

import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for sequence features.
    Applies rotation to pairs of dimensions based on position.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len: int, offset: int = 0):
        """
        Generate rotary embeddings for positions [offset, offset+seq_len).
        Returns: (seq_len, dim//2, 2) cos/sin pairs reshaped for broadcasting.
        """
        t = torch.arange(offset, offset + seq_len, 
                        device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """
    Apply rotary position embeddings to query and key tensors.
    Args:
        q: (B, h, L, d) query
        k: (B, h, L, d) key
        cos: (L, d) cosine
        sin: (L, d) sine
    Returns:
        q, k with rotary embeddings applied
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (B, L, D) input sequence
            mask: (B, L) optional attention mask
        Returns:
            out: (B, L, D) attention output
        """
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, L, d)
        q, k, v = qkv.unbind(0)

        # Apply rotary embeddings
        cos, sin = self.rope(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, L)

        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            attn = attn.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0, 
                float("-inf")
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class ContextTransformerBlock(nn.Module):
    """Single Transformer block with RoPE and pre-LN."""

    def __init__(self, dim: int, num_heads: int = 8, 
                 ff_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.attn = RoPEAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ff_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ff_ratio), dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Pre-LN Transformer block."""
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ContextTransformer(nn.Module):
    """
    K-layer Transformer with Rotary Position Encoding
    for modality-specific context enhancement.
    """

    def __init__(self, dim: int, num_layers: int = 4, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ContextTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence features
            mask: (B, L) optional attention mask (0 = masked)
        Returns:
            x: (B, L, D) context-enhanced features
        """
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
