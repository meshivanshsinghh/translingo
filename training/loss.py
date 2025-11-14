import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, 
                 padding_idx: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch_size * seq_len, vocab_size]
            target: Target indices [batch_size * seq_len]
        
        Returns:
            Loss value
        """
        batch_size = pred.size(0)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            
            # Mask padding tokens
            mask = target == self.padding_idx
            if mask.any():
                true_dist.index_fill_(0, mask.nonzero(as_tuple=True)[0], 0.0)
        
        # Calculate KL divergence
        loss = F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='none')
        loss = loss.sum(dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            # Mean over non-padding tokens
            non_pad_mask = target != self.padding_idx
            loss = loss.sum() / non_pad_mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class TransformerLoss(nn.Module):
    """
    Combined loss function for transformer training
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1,
                 padding_idx: int = 0):
        super().__init__()
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            smoothing=smoothing,
            padding_idx=padding_idx,
            reduction='mean'
        )
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Model predictions [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]
            mask: Optional mask [batch_size, seq_len]
        
        Returns:
            Loss value
        """
        # Reshape for loss calculation
        pred = pred.contiguous().view(-1, pred.size(-1))
        target = target.contiguous().view(-1)
        
        # Calculate loss
        loss = self.criterion(pred, target)
        
        return loss
