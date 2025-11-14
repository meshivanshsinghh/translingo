import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from typing import List, Tuple, Optional, Dict
import yaml
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], tokenizer_path: str, 
                 max_length: int = 100, config_path: str = "configs/config.yaml"):
        """
        Translation dataset for German-English pairs
        
        Args:
            data: List of (source, target) text pairs
            tokenizer_path: Path to SentencePiece model
            max_length: Maximum sequence length
            config_path: Path to config file
        """
        self.data = data
        self.max_length = max_length
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # Special tokens
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        logger.info(f"Vocab size: {self.sp.vocab_size()}")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text, tgt_text = self.data[idx]
        
        # Tokenize
        src_tokens = self.sp.encode(src_text, out_type=int)
        tgt_tokens = self.sp.encode(tgt_text, out_type=int)
        
        # Truncate if necessary
        src_tokens = src_tokens[:self.max_length - 2]  # Leave room for BOS/EOS
        tgt_tokens = tgt_tokens[:self.max_length - 2]
        
        # Add BOS and EOS tokens
        src_tokens = [self.bos_id] + src_tokens + [self.eos_id]
        tgt_tokens = [self.bos_id] + tgt_tokens + [self.eos_id]
        
        # Convert to tensors
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'src_len': len(src_tokens),
            'tgt_len': len(tgt_tokens)
        }


class DataCollator:
    def __init__(self, pad_id: int = 0):
        """
        Collator for batching translation data
        
        Args:
            pad_id: Padding token ID
        """
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract sequences
        src_seqs = [item['src'] for item in batch]
        tgt_seqs = [item['tgt'] for item in batch]
        
        # Pad sequences
        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=self.pad_id)
        
        # Create attention masks (1 for real tokens, 0 for padding)
        src_mask = (src_padded != self.pad_id).float()
        tgt_mask = (tgt_padded != self.pad_id).float()
        
        return {
            'src': src_padded,
            'tgt': tgt_padded,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask
        }


def create_dataloaders(train_data: List[Tuple[str, str]], 
                      valid_data: List[Tuple[str, str]], 
                      test_data: List[Tuple[str, str]],
                      tokenizer_path: str,
                      batch_size: int = 32,
                      num_workers: int = 2,
                      config_path: str = "configs/config.yaml") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_data: Training data
        valid_data: Validation data
        test_data: Test data
        tokenizer_path: Path to tokenizer model
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        config_path: Path to config file
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = TranslationDataset(train_data, tokenizer_path, config_path=config_path)
    valid_dataset = TranslationDataset(valid_data, tokenizer_path, config_path=config_path)
    test_dataset = TranslationDataset(test_data, tokenizer_path, config_path=config_path)
    
    # Create collator
    collator = DataCollator(pad_id=train_dataset.pad_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


def analyze_dataset(data: List[Tuple[str, str]], tokenizer_path: str) -> Dict:
    """
    Analyze dataset statistics
    
    Args:
        data: List of (source, target) pairs
        tokenizer_path: Path to tokenizer
    
    Returns:
        Dictionary with statistics
    """
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    src_lengths = []
    tgt_lengths = []
    
    for src, tgt in data:
        src_tokens = sp.encode(src)
        tgt_tokens = sp.encode(tgt)
        src_lengths.append(len(src_tokens))
        tgt_lengths.append(len(tgt_tokens))
    
    stats = {
        'num_samples': len(data),
        'src_avg_length': np.mean(src_lengths),
        'src_max_length': np.max(src_lengths),
        'src_min_length': np.min(src_lengths),
        'tgt_avg_length': np.mean(tgt_lengths),
        'tgt_max_length': np.max(tgt_lengths),
        'tgt_min_length': np.min(tgt_lengths),
        'vocab_size': sp.vocab_size()
    }
    
    return stats


if __name__ == "__main__":
    # Test the dataset
    from data.download import DataDownloader
    
    downloader = DataDownloader()
    train_data, valid_data, test_data = downloader.download_multi30k()
    
    if train_data:
        tokenizer_path = os.path.join('data', 'processed', 'tokenizer.model')
        
        # Analyze dataset
        stats = analyze_dataset(train_data, tokenizer_path)
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Create dataloaders
        train_loader, valid_loader, test_loader = create_dataloaders(
            train_data[:100],  # Use small subset for testing
            valid_data[:10],
            test_data[:10],
            tokenizer_path,
            batch_size=8
        )
        
        # Test loading a batch
        for batch in train_loader:
            logger.info(f"Batch shapes:")
            logger.info(f"  src: {batch['src'].shape}")
            logger.info(f"  tgt: {batch['tgt'].shape}")
            logger.info(f"  src_mask: {batch['src_mask'].shape}")
            logger.info(f"  tgt_mask: {batch['tgt_mask'].shape}")
            break
