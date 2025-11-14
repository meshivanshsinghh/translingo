import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism with numerical stability"""
    
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
        # Calculate attention scores with temperature scaling
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.temperature * math.sqrt(d_k))
        
        # Apply mask if provided - using fp16-safe value
        if mask is not None:
            # Determine safe mask value based on dtype
            if scores.dtype == torch.float16:
                mask_value = -65504.0  # Max negative value for fp16
            else:
                mask_value = -1e9  # Original value for fp32
            
            # Use torch.finfo for more robust dtype handling
            mask_value = torch.finfo(scores.dtype).min if hasattr(torch, 'finfo') else mask_value
            scores = scores.masked_fill(mask == 0, mask_value)
        
        # Apply softmax with numerical stability
        attention = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with improved stability"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_bias: bool = True, pre_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pre_norm = pre_norm
        
        # Linear projections with optional bias
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Initialize weights using Xavier uniform
        self._init_weights()
        
        # Attention
        self.attention = ScaledDotProductAttention(temperature=1.0, dropout=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Mask tensor
        
        Returns:
            output: Multi-head attention output [batch_size, seq_len_q, d_model]
            attention: Attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)  # Query sequence length
        seq_len_k = key.size(1)     # Key sequence length (can be different!)
        seq_len_v = value.size(1)   # Value sequence length (same as key)
        
        # Pre-norm variant (if enabled)
        if self.pre_norm:
            query = self.layer_norm(query)
            key = self.layer_norm(key)
            value = self.layer_norm(value)
        
        # Store residual
        residual = query
        
        # Linear projections - FIXED: Use correct sequence lengths
        Q = self.W_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads - use seq_len_q for output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # Add residual and normalize
        output = output + residual
        if not self.pre_norm:
            output = self.layer_norm(output)
        
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
    # Create boolean mask
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(torch.bool)


def create_look_ahead_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        size: Sequence length
        device: Device to create mask on
    
    Returns:
        mask: Look-ahead mask [1, 1, size, size]
    """
    # Create upper triangular matrix
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    # Invert it (1 for allowed positions, 0 for masked)
    mask = ~mask
    return mask.unsqueeze(0).unsqueeze(0)


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
    
    # Target padding mask
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    
    # Target look-ahead mask
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len, tgt.device)
    
    # Combine padding and look-ahead masks for target
    # Both masks should be True where attention is allowed
    tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
    
    # Memory mask (same as source mask)
    memory_mask = src_mask
    
    return src_mask, tgt_mask, memory_mask


# Optional: Flash Attention wrapper (if available)
try:
    from torch.nn.functional import scaled_dot_product_attention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

class FlashAttention(nn.Module):
    """Flash Attention wrapper for better performance (if available)"""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """
        Uses PyTorch's scaled_dot_product_attention if available (includes Flash Attention)
        """
        if FLASH_ATTENTION_AVAILABLE and mask is None:
            # Use efficient implementation when no mask
            output = scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            return output, None
        else:
            # Fallback to standard implementation
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                mask_value = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask == 0, mask_value)
            
            attention = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attention = F.dropout(attention, p=self.dropout)
            
            output = torch.matmul(attention, v)
            return output, attention