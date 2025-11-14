import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from model.embeddings import TransformerEmbedding
from model.layers import Encoder, Decoder
from model.attention import create_masks

class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 1024, max_seq_length: int = 100,
                 dropout: float = 0.1, pad_idx: int = 0, share_embeddings: bool = True):
        super().__init__()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        if share_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # Encoder and Decoder
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Share weights between embedding and output projection if specified
        if share_embeddings:
            self.output_projection.weight = self.src_embedding.token_embedding.embedding.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
        
        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """
        # Create masks
        src_mask, tgt_mask, memory_mask = create_masks(src, tgt, self.pad_idx)
        
        # Encode source
        src_emb = self.src_embedding(src)
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode target
        tgt_emb = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, memory_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_mask: Source mask
        
        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        if src_mask is None:
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        src_emb = self.src_embedding(src)
        encoder_output = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence given encoder output
        
        Args:
            tgt: Target sequence [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target mask
            memory_mask: Memory mask
        
        Returns:
            Decoder output [batch_size, tgt_len, d_model]
        """
        tgt_emb = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        
        return decoder_output
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_length: int = 100,
                 bos_id: int = 2, eos_id: int = 3,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate translation using greedy decoding
        
        Args:
            src: Source sequence [batch_size, src_len]
            max_length: Maximum generation length
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            temperature: Sampling temperature
        
        Returns:
            Generated sequences [batch_size, seq_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        memory = self.encode(src, src_mask)
        
        # Initialize target with BOS token
        tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Create masks
            tgt_mask = torch.ones(batch_size, 1, tgt.size(1), tgt.size(1), device=device)
            tgt_mask = torch.tril(tgt_mask)
            
            # Decode
            tgt_emb = self.tgt_embedding(tgt)
            decoder_output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
            
            # Get next token logits
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Greedy selection
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_id).all():
                break
        
        return tgt
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if non_embedding:
            # Subtract embedding parameters
            emb_params = sum(p.numel() for name, p in self.named_parameters() 
                           if 'embedding' in name and p.requires_grad)
            params -= emb_params
        
        return params


def create_model(config: Dict) -> Transformer:
    """
    Create transformer model from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Transformer model
    """
    return Transformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout']
    )
