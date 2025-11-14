import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for sin/cos frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding with scaling"""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices [batch_size, seq_len]
        
        Returns:
            Scaled embeddings [batch_size, seq_len, d_model]
        """
        return self.embedding(x) * self.scale


class TransformerEmbedding(nn.Module):
    """Combined token and positional embedding for transformer"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, 
                 dropout: float = 0.1, scale_embedding: bool = True):
        super().__init__()
        
        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model) if scale_embedding else nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Optional learned positional embeddings (alternative to sinusoidal)
        self.use_learned_pos = False
        if self.use_learned_pos:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input token indices [batch_size, seq_len]
            pos: Optional position indices for learned positional embeddings
        
        Returns:
            Embedded and encoded tensor [batch_size, seq_len, d_model]
        """
        # Get token embeddings
        if isinstance(self.token_embedding, TokenEmbedding):
            token_emb = self.token_embedding(x)
        else:
            token_emb = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        
        # Add positional encoding
        if self.use_learned_pos and pos is not None:
            pos_emb = self.pos_embedding(pos)
            output = token_emb + pos_emb
            output = self.positional_encoding.dropout(output)
        else:
            output = self.positional_encoding(token_emb)
        
        return output


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (alternative to sinusoidal)"""
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with learned positional embeddings added
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Add positional embeddings
        x = x + self.embedding(positions)
        
        return self.dropout(x)
