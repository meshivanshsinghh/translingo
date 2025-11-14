import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism"""
    
    def __init__(self, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch_size, n_heads, seq_len, d_k]
            k: Key tensor [batch_size, n_heads, seq_len, d_k]
            v: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Mask tensor [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        
        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_k]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.temperature * math.sqrt(q.size(-1)))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention
        self.attention = ScaledDotProductAttention(temperature=1.0, dropout=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Mask tensor
        
        Returns:
            output: Multi-head attention output [batch_size, seq_len, d_model]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Store residual
        residual = query
        
        # Linear projections in batch from d_model => n_heads x d_k
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # Add and normalize
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding index
    
    Returns:
        mask: Padding mask [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        size: Sequence length
        device: Device to create mask on
    
    Returns:
        mask: Look-ahead mask [1, 1, size, size]
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return (1 - mask).unsqueeze(0).unsqueeze(0)


def create_masks(src: torch.Tensor, tgt: torch.Tensor, 
                 pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create all masks needed for transformer
    
    Args:
        src: Source sequence [batch_size, src_len]
        tgt: Target sequence [batch_size, tgt_len]
        pad_idx: Padding index
    
    Returns:
        src_mask: Source padding mask
        tgt_mask: Target mask (padding + look-ahead)
        memory_mask: Memory mask for decoder cross-attention
    """
    # Source mask (padding only)
    src_mask = create_padding_mask(src, pad_idx)
    
    # Target mask (padding + look-ahead)
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len, tgt.device)
    tgt_mask = tgt_pad_mask.float() * tgt_look_ahead_mask.float()
    tgt_mask = tgt_mask.bool()
    
    # Memory mask (same as source mask but different shape)
    memory_mask = src_mask
    
    return src_mask, tgt_mask, memory_mask
