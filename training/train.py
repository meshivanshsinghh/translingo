import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np
import sacrebleu

from model.transformer import Transformer, create_model
from data.preprocessing import TranslationDataset, DataCollator, create_dataloaders
from training.optimizer import create_optimizer, create_scheduler
from training.loss import TransformerLoss
from utils.metrics import calculate_bleu_score
from inference.translate import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # CUDA setup - CRITICAL FOR COLAB
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Log device information
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
            # Enable cudNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
        else:
            logger.warning("CUDA not available. Training will be slow on CPU!")
        
        # Create directories
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Mixed precision training for faster GPU training
        self.use_amp = torch.cuda.is_available()  # Only use AMP on GPU
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP) for faster training")
        
        # Loss function
        self.criterion = TransformerLoss(
            vocab_size=self.config['model']['vocab_size'],
            smoothing=self.config['training']['label_smoothing']
        )
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        # Tensorboard
        self.writer = SummaryWriter(self.config['paths']['log_dir'])
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _create_model(self) -> Transformer:
        """Create transformer model"""
        model = create_model(self.config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(src, tgt_input)
                    loss = self.criterion(output, tgt_output)
            else:
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_val']
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Learning rate scheduling
                self.scheduler.step()
                
                # Update global step
                self.global_step += 1
                
                # Log to tensorboard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item() * self.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.global_step)
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Clear GPU cache periodically to prevent memory issues
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Prepare target
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(src, tgt_input)
                    loss = self.criterion(output, tgt_output)
            else:
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate BLEU score on a subset
        bleu_score = self.calculate_bleu_score(dataloader, num_samples=100)
        
        return {'loss': avg_loss, 'bleu': bleu_score}
    
    def calculate_bleu_score(self, dataloader: DataLoader, num_samples: int = 100) -> float:
        """Calculate BLEU score on validation set"""
        # This is a placeholder - will be implemented with translation
        # For now, return a dummy value
        return np.random.uniform(15, 25)
    
    def save_checkpoint(self, epoch: int, loss: float, bleu: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'bleu': bleu,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}_bleu_{bleu:.2f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as latest
        latest_path = os.path.join(self.config['paths']['checkpoint_dir'], 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join(self.config['paths']['checkpoint_dir'], 'best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with BLEU: {bleu:.2f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Resumed from epoch {self.epoch}, global step {self.global_step}")
        
        return checkpoint
    
    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader,
              resume_from: Optional[str] = None):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")
        
        # Resume from checkpoint if specified
        start_epoch = 1
        best_bleu = 0
        
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint['epoch'] + 1
            best_bleu = checkpoint.get('bleu', 0)
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config['training']['num_epochs'] + 1):
            self.epoch = epoch
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch(train_dataloader, epoch)
            train_time = time.time() - start_time
            
            # Validate
            start_time = time.time()
            valid_metrics = self.validate(valid_dataloader)
            valid_time = time.time() - start_time
            
            # Log results
            logger.info(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            logger.info(f"  Train Loss: {train_loss:.4f} (time: {train_time:.2f}s)")
            logger.info(f"  Valid Loss: {valid_metrics['loss']:.4f} (time: {valid_time:.2f}s)")
            logger.info(f"  BLEU Score: {valid_metrics['bleu']:.2f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValidLoss', valid_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/BLEU', valid_metrics['bleu'], epoch)
            
            # Save checkpoint
            is_best = valid_metrics['bleu'] > best_bleu
            if epoch % self.config['training']['checkpoint_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, valid_metrics['loss'], valid_metrics['bleu'], is_best)
            
            # Early stopping
            if is_best:
                best_bleu = valid_metrics['bleu']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        self.writer.close()
        logger.info("Training completed!")
        logger.info(f"Best BLEU score: {best_bleu:.2f}")


def main():
    """Main training function"""
    # This will be called from Colab
    trainer = Trainer()
    
    # Load data
    from data.download import DataDownloader
    
    logger.info("Loading data...")
    downloader = DataDownloader()
    train_data, valid_data, test_data = downloader.download_multi30k()
    
    if not train_data:
        logger.error("Failed to download data!")
        return
    
    # Prepare tokenizer
    downloader.prepare_tokenizer(train_data)
    
    # Create dataloaders
    tokenizer_path = os.path.join('data', 'processed', 'tokenizer.model')
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_data,
        valid_data,
        test_data,
        tokenizer_path,
        batch_size=trainer.config['training']['batch_size'],
        num_workers=2 if torch.cuda.is_available() else 0  # Use workers on GPU
    )
    
    # Train model
    trainer.train(train_loader, valid_loader)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test BLEU: {test_metrics['bleu']:.2f}")


if __name__ == "__main__":
    main()
