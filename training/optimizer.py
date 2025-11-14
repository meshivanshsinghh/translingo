import torch.optim as optim
import numpy as np
from typing import Optional, List

class NoamScheduler:
    """
    Noam learning rate scheduler as described in "Attention is All You Need"
    """
    def __init__(self, optimizer: optim.Optimizer, d_model: int, 
                 warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0
        
        # Store initial learning rates
        self._initial_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self) -> float:
        """Calculate learning rate based on current step"""
        arg1 = self.current_step ** (-0.5)
        arg2 = self.current_step * (self.warmup_steps ** (-1.5))
        
        return self.factor * (self.d_model ** (-0.5)) * min(arg1, arg2)
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> dict:
        """Get scheduler state"""
        return {
            'current_step': self.current_step,
            'factor': self.factor,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state"""
        self.current_step = state_dict['current_step']
        self.factor = state_dict.get('factor', 1.0)
        self.d_model = state_dict.get('d_model', self.d_model)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)


class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate scheduler with warmup
    """
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # Store initial learning rates
        self._initial_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self._initial_lrs[0])
        
        for param_group, initial_lr in zip(self.optimizer.param_groups, self._initial_lrs):
            param_group['lr'] = initial_lr * lr_scale
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> dict:
        """Get scheduler state"""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state"""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.min_lr = state_dict.get('min_lr', self.min_lr)


def create_optimizer(model, config: dict) -> optim.Optimizer:
    """
    Create optimizer from config
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
    
    Returns:
        Optimizer instance
    """
    lr = config['training']['learning_rate']
    
    # Adam optimizer with specific betas for transformer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """
    Create learning rate scheduler from config
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
    
    Returns:
        Scheduler instance
    """
    d_model = config['model']['d_model']
    warmup_steps = config['training']['warmup_steps']
    
    # Use Noam scheduler by default
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps
    )
    
    return scheduler
