import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class BeamHypothesis:
    """Single hypothesis in beam search"""
    tokens: List[int]
    log_prob: float
    finished: bool = False

class BeamSearch:
    """Beam search decoder for transformer models"""
    
    def __init__(self, beam_size: int = 4, length_penalty: float = 0.6,
                 coverage_penalty: float = 0.0, no_repeat_ngram_size: int = 0):
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
    
    def search(self, model, src: torch.Tensor, max_length: int = 100,
               bos_id: int = 2, eos_id: int = 3, pad_id: int = 0) -> List[List[int]]:
        """
        Perform beam search decoding
        
        Args:
            model: Transformer model
            src: Source sequence [batch_size, src_len]
            max_length: Maximum decoding length
            bos_id: Beginning of sequence token
            eos_id: End of sequence token
            pad_id: Padding token
        
        Returns:
            List of decoded sequences
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
        memory = model.encode(src, src_mask)
        
        # Initialize beams
        beams = [[BeamHypothesis([bos_id], 0.0)] for _ in range(batch_size)]
        
        for step in range(max_length - 1):
            all_candidates = []
            
            for batch_idx in range(batch_size):
                # Skip if all beams are finished
                if all(hyp.finished for hyp in beams[batch_idx]):
                    continue
                
                # Prepare input for all beams
                beam_tokens = []
                beam_indices = []
                
                for beam_idx, hypothesis in enumerate(beams[batch_idx]):
                    if not hypothesis.finished:
                        beam_tokens.append(hypothesis.tokens)
                        beam_indices.append(beam_idx)
                
                if not beam_tokens:
                    continue
                
                # Create batch of sequences
                tgt = torch.tensor(beam_tokens, device=device)
                
                # Decode
                tgt_mask = torch.ones(len(beam_tokens), 1, tgt.size(1), tgt.size(1), device=device)
                tgt_mask = torch.tril(tgt_mask)
                
                # Expand memory for beam size
                expanded_memory = memory[batch_idx:batch_idx+1].expand(len(beam_tokens), -1, -1)
                expanded_src_mask = src_mask[batch_idx:batch_idx+1].expand(len(beam_tokens), -1, -1, -1)
                
                # Get predictions
                decoder_output = model.decode(tgt, expanded_memory, tgt_mask, expanded_src_mask)
                logits = model.output_projection(decoder_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k tokens for each beam
                vocab_size = log_probs.size(-1)
                top_log_probs, top_indices = torch.topk(log_probs, min(self.beam_size, vocab_size))
                
                # Create new candidates
                candidates = []
                
                for beam_local_idx, (beam_idx, beam_log_probs, beam_indices_local) in enumerate(
                    zip(beam_indices, top_log_probs, top_indices)):
                    
                    hypothesis = beams[batch_idx][beam_idx]
                    
                    for token_rank, (token_log_prob, token_id) in enumerate(
                        zip(beam_log_probs, beam_indices_local)):
                        
                        # Apply no-repeat penalty
                        if self._has_repeated_ngram(hypothesis.tokens + [token_id.item()]):
                            continue
                        
                        new_log_prob = hypothesis.log_prob + token_log_prob.item()
                        
                        # Apply length penalty
                        score = self._apply_length_penalty(new_log_prob, len(hypothesis.tokens) + 1)
                        
                        candidates.append((
                            score,
                            BeamHypothesis(
                                tokens=hypothesis.tokens + [token_id.item()],
                                log_prob=new_log_prob,
                                finished=(token_id.item() == eos_id)
                            )
                        ))
                
                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                new_beams = []
                for score, hypothesis in candidates[:self.beam_size]:
                    new_beams.append(hypothesis)
                
                beams[batch_idx] = new_beams
        
        # Extract best sequences
        results = []
        for batch_idx in range(batch_size):
            # Sort by score
            sorted_hyps = sorted(
                beams[batch_idx],
                key=lambda h: self._apply_length_penalty(h.log_prob, len(h.tokens)),
                reverse=True
            )
            
            # Get best hypothesis
            best_hyp = sorted_hyps[0]
            results.append(best_hyp.tokens)
        
        return results
    
    def _apply_length_penalty(self, log_prob: float, length: int) -> float:
        """Apply length penalty to score"""
        return log_prob / (length ** self.length_penalty)
    
    def _has_repeated_ngram(self, tokens: List[int]) -> bool:
        """Check if sequence has repeated n-grams"""
        if self.no_repeat_ngram_size <= 0:
            return False
        
        ngrams = set()
        for i in range(len(tokens) - self.no_repeat_ngram_size + 1):
            ngram = tuple(tokens[i:i + self.no_repeat_ngram_size])
            if ngram in ngrams:
                return True
            ngrams.add(ngram)
        
        return False


class GreedyDecoder:
    """Simple greedy decoder for fast inference"""
    
    @staticmethod
    def decode(model, src: torch.Tensor, max_length: int = 100,
               bos_id: int = 2, eos_id: int = 3, pad_id: int = 0) -> List[List[int]]:
        """
        Perform greedy decoding
        
        Args:
            model: Transformer model
            src: Source sequence [batch_size, src_len]
            max_length: Maximum decoding length
            bos_id: Beginning of sequence token
            eos_id: End of sequence token
            pad_id: Padding token
        
        Returns:
            List of decoded sequences
        """
        batch_size = src.size(0)
        device = src.device
        
        # Use model's built-in generate method
        with torch.no_grad():
            translations = model.generate(
                src, 
                max_length=max_length,
                bos_id=bos_id,
                eos_id=eos_id
            )
        
        # Convert to list
        results = []
        for i in range(batch_size):
            tokens = translations[i].cpu().tolist()
            # Remove padding and special tokens if needed
            if eos_id in tokens:
                eos_idx = tokens.index(eos_id)
                tokens = tokens[:eos_idx + 1]
            results.append(tokens)
        
        return results
