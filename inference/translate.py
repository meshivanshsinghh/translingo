import torch
import sentencepiece as spm
from typing import List, Optional, Dict, Tuple
import logging
from model.transformer import Transformer
from inference.beam_search import BeamSearch, GreedyDecoder

logger = logging.getLogger(__name__)

class Translator:
    """High-level translation interface"""
    
    def __init__(self, model: Transformer, tokenizer_path: str, 
                 device: Optional[torch.device] = None,
                 beam_size: int = 4, use_beam_search: bool = True):
        """
        Initialize translator
        
        Args:
            model: Trained transformer model
            tokenizer_path: Path to sentencepiece model
            device: Device to run on
            beam_size: Beam size for beam search
            use_beam_search: Whether to use beam search or greedy decoding
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # Special tokens
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        
        # Decoder - with no_repeat_ngram_size=3 to prevent repetition
        self.use_beam_search = use_beam_search
        if use_beam_search:
            self.decoder = BeamSearch(beam_size=beam_size, no_repeat_ngram_size=3)
        
        logger.info(f"Translator initialized on {self.device}")
        logger.info(f"Vocab size: {self.sp.vocab_size()}")
        logger.info(f"Using {'beam search' if use_beam_search else 'greedy'} decoding")
    
    def translate(self, text: str, max_length: int = 50) -> str:  # Changed default from 100 to 50
        """
        Translate a single text
        
        Args:
            text: Source text to translate
            max_length: Maximum translation length
        
        Returns:
            Translated text
        """
        # Tokenize
        tokens = self.sp.encode(text)
        
        # Add special tokens
        tokens = [self.bos_id] + tokens + [self.eos_id]
        
        # Convert to tensor
        src = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Decode
        with torch.no_grad():
            if self.use_beam_search:
                translations = self.decoder.search(
                    self.model, src, max_length,
                    self.bos_id, self.eos_id, self.pad_id
                )
            else:
                translations = GreedyDecoder.decode(
                    self.model, src, max_length,
                    self.bos_id, self.eos_id, self.pad_id
                )
        
        # Decode tokens
        translated_tokens = translations[0]
        
        # Remove special tokens
        if self.bos_id in translated_tokens:
            translated_tokens = translated_tokens[translated_tokens.index(self.bos_id) + 1:]
        if self.eos_id in translated_tokens:
            translated_tokens = translated_tokens[:translated_tokens.index(self.eos_id)]
        
        # Decode to text
        translated_text = self.sp.decode(translated_tokens)
        
        return translated_text
    
    def translate_batch(self, texts: List[str], max_length: int = 50) -> List[str]:  # Changed from 100 to 50
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of source texts
            max_length: Maximum translation length
        
        Returns:
            List of translated texts
        """
        # Tokenize all texts
        tokenized = []
        for text in texts:
            tokens = self.sp.encode(text)
            tokens = [self.bos_id] + tokens + [self.eos_id]
            tokenized.append(tokens)
        
        # Pad sequences
        max_len = max(len(tokens) for tokens in tokenized)
        padded = []
        for tokens in tokenized:
            padded_tokens = tokens + [self.pad_id] * (max_len - len(tokens))
            padded.append(padded_tokens)
        
        # Convert to tensor
        src = torch.tensor(padded, dtype=torch.long).to(self.device)
        
        # Decode
        with torch.no_grad():
            if self.use_beam_search:
                translations = self.decoder.search(
                    self.model, src, max_length,
                    self.bos_id, self.eos_id, self.pad_id
                )
            else:
                translations = GreedyDecoder.decode(
                    self.model, src, max_length,
                    self.bos_id, self.eos_id, self.pad_id
                )
        
        # Decode all translations
        results = []
        for translated_tokens in translations:
            # Remove special tokens
            if self.bos_id in translated_tokens:
                translated_tokens = translated_tokens[translated_tokens.index(self.bos_id) + 1:]
            if self.eos_id in translated_tokens:
                translated_tokens = translated_tokens[:translated_tokens.index(self.eos_id)]
            
            # Decode to text
            translated_text = self.sp.decode(translated_tokens)
            results.append(translated_text)
        
        return results
    
    def translate_with_attention(self, text: str, max_length: int = 50) -> Tuple[str, torch.Tensor]:
        """
        Translate and return attention weights
        
        Args:
            text: Source text to translate
            max_length: Maximum translation length
        
        Returns:
            Tuple of (translated_text, attention_weights)
        """
        # This is a placeholder - would need to modify model to return attention
        translation = self.translate(text, max_length)
        
        # For now, return dummy attention
        src_len = len(self.sp.encode(text)) + 2  # +2 for BOS/EOS
        tgt_len = len(self.sp.encode(translation)) + 2
        attention = torch.rand(1, self.model.n_heads, tgt_len, src_len)
        
        return translation, attention
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str,
                       device: Optional[torch.device] = None, **kwargs):
        """
        Load translator from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            device: Device to load on
            **kwargs: Additional arguments for translator
        
        Returns:
            Translator instance
        """
        # Load checkpoint
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        config = checkpoint['config']
        model = Transformer(
            vocab_size=config['model']['vocab_size'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            d_ff=config['model']['d_ff'],
            max_seq_length=config['model']['max_seq_length'],
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create translator
        return cls(model, tokenizer_path, device, **kwargs)


def interactive_translation(checkpoint_path: str, tokenizer_path: str):
    """
    Interactive translation in terminal
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
    """
    # Load translator
    translator = Translator.from_checkpoint(checkpoint_path, tokenizer_path)
    
    print("TransLingo Interactive Translation")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get input
        text = input("\nEnter German text: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        # Translate
        try:
            translation = translator.translate(text)
            print(f"English translation: {translation}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python translate.py <checkpoint_path> <tokenizer_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    
    interactive_translation(checkpoint_path, tokenizer_path)