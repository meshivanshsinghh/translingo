import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Store residual
        residual = x
        
        # Feed-forward network
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Add and normalize
        x = self.layer_norm(x + residual)
        
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention
        x, _ = self.self_attention(x, x, x, mask)
        
        # Feed-forward
        x = self.feed_forward(x)
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
        
        Returns:
            output: Output tensor [batch_size, tgt_len, d_model]
            self_attn: Self-attention weights
            cross_attn: Cross-attention weights
        """
        # Masked self-attention
        x, self_attn = self.self_attention(x, x, x, tgt_mask)
        
        # Cross-attention
        x, cross_attn = self.cross_attention(x, memory, memory, memory_mask)
        
        # Feed-forward
        x = self.feed_forward(x)
        
        return x, self_attn, cross_attn


class Encoder(nn.Module):
    """Transformer encoder"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x


class Decoder(nn.Module):
    """Transformer decoder"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
        
        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Pass through decoder layers
        for layer in self.layers:
            x, _, _ = layer(x, memory, tgt_mask, memory_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        return x
