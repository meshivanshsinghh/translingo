import torch
import numpy as np
from typing import List, Tuple, Optional
import sacrebleu
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def calculate_bleu_score(predictions: List[str], references: List[str], 
                        smooth_method: str = 'exp') -> float:
    """
    Calculate BLEU score using sacrebleu
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        smooth_method: Smoothing method for BLEU
    
    Returns:
        BLEU score
    """
    if not predictions or not references:
        return 0.0
    
    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]
    
    try:
        bleu = sacrebleu.corpus_bleu(predictions, refs, smooth_method=smooth_method)
        return bleu.score
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return 0.0


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity
    """
    return np.exp(loss)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                      pad_idx: int = 0) -> float:
    """
    Calculate token-level accuracy
    
    Args:
        predictions: Predicted token indices [batch_size, seq_len]
        targets: Target token indices [batch_size, seq_len]
        pad_idx: Padding token index to ignore
    
    Returns:
        Accuracy percentage
    """
    # Create mask for non-padding tokens
    mask = targets != pad_idx
    
    # Calculate correct predictions
    correct = (predictions == targets) & mask
    
    # Calculate accuracy
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item() * 100


def calculate_token_diversity(sentences: List[List[int]]) -> float:
    """
    Calculate token diversity (unique tokens / total tokens)
    
    Args:
        sentences: List of tokenized sentences
    
    Returns:
        Diversity score
    """
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(sent)
    
    if not all_tokens:
        return 0.0
    
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    return unique_tokens / total_tokens


def calculate_length_ratio(predictions: List[List[int]], 
                          references: List[List[int]]) -> float:
    """
    Calculate average length ratio between predictions and references
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
    
    Returns:
        Average length ratio
    """
    if not predictions or not references:
        return 0.0
    
    ratios = []
    for pred, ref in zip(predictions, references):
        if len(ref) > 0:
            ratios.append(len(pred) / len(ref))
    
    return np.mean(ratios) if ratios else 0.0


class MetricTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, name: str, value: float, count: int = 1):
        """Update metric with new value"""
        if name not in self.metrics:
            self.metrics[name] = {'sum': 0.0, 'count': 0}
        
        self.metrics[name]['sum'] += value * count
        self.metrics[name]['count'] += count
    
    def get_average(self, name: str) -> float:
        """Get average value for metric"""
        if name not in self.metrics or self.metrics[name]['count'] == 0:
            return 0.0
        
        return self.metrics[name]['sum'] / self.metrics[name]['count']
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
    
    def get_all_averages(self) -> dict:
        """Get all average metrics"""
        return {name: self.get_average(name) for name in self.metrics}


def evaluate_translation_quality(predictions: List[str], references: List[str]) -> dict:
    """
    Comprehensive evaluation of translation quality
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # BLEU scores with different n-grams
    for n in range(1, 5):
        refs = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs, force=True, 
                                     lowercase=False, tokenize='none',
                                     smooth_method='exp', max_ngram_order=n)
        metrics[f'bleu-{n}'] = bleu.score
    
    # Overall BLEU
    metrics['bleu'] = calculate_bleu_score(predictions, references)
    
    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    
    metrics['avg_pred_length'] = np.mean(pred_lengths)
    metrics['avg_ref_length'] = np.mean(ref_lengths)
    metrics['length_ratio'] = metrics['avg_pred_length'] / metrics['avg_ref_length']
    
    return metrics
