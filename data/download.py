import os
import torch
import sentencepiece as spm
from typing import List, Tuple, Optional, Dict
import yaml
import logging
from tqdm import tqdm
import urllib.request

try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

try:
    from torchtext.datasets import Multi30k
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    TORCHTEXT_AVAILABLE = True
except Exception as e:
    TORCHTEXT_AVAILABLE = False
    print(f"Warning: torchtext import failed: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['paths']['data_dir']
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        
    def download_multi30k(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Download Multi30k dataset - tries multiple methods"""
        logger.info("Downloading Multi30k dataset...")
        
        # Method 1: Try Hugging Face first (most reliable)
        if HUGGINGFACE_AVAILABLE:
            try:
                logger.info("Attempting download from Hugging Face...")
                return self._download_from_huggingface()
            except Exception as e:
                logger.warning(f"Hugging Face download failed: {e}")
        
        # Method 2: Try torchtext if available
        if TORCHTEXT_AVAILABLE:
            try:
                logger.info("Attempting download with torchtext...")
                train_data = list(Multi30k(split='train', language_pair=('de', 'en')))
                valid_data = list(Multi30k(split='valid', language_pair=('de', 'en')))
                test_data = list(Multi30k(split='test', language_pair=('de', 'en')))
                
                logger.info(f"Train samples: {len(train_data)}")
                logger.info(f"Valid samples: {len(valid_data)}")
                logger.info(f"Test samples: {len(test_data)}")
                
                self._save_data_to_files(train_data, valid_data, test_data)
                return train_data, valid_data, test_data
            except Exception as e:
                logger.warning(f"Torchtext download failed: {e}")
        
        # Method 3: Try manual download from GitHub
        logger.info("Attempting manual download from GitHub...")
        return self._download_multi30k_manual()
    
    def _download_from_huggingface(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Download Multi30k from Hugging Face datasets hub"""
        logger.info("Downloading from Hugging Face datasets hub...")
        
        # Load dataset
        dataset = load_dataset("bentrevett/multi30k")
        
        # Convert to expected format: List[Tuple[str, str]]
        train_data = [(item['de'], item['en']) for item in dataset['train']]
        valid_data = [(item['de'], item['en']) for item in dataset['validation']]
        test_data = [(item['de'], item['en']) for item in dataset['test']]
        
        logger.info(f"✅ Downloaded from Hugging Face:")
        logger.info(f"  Train samples: {len(train_data)}")
        logger.info(f"  Valid samples: {len(valid_data)}")
        logger.info(f"  Test samples: {len(test_data)}")
        
        # Save to files for consistency with other methods
        self._save_data_to_files(train_data, valid_data, test_data)
        
        return train_data, valid_data, test_data
    
    def _download_multi30k_manual(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Manual download of Multi30k dataset from GitHub"""
        # Try multiple mirror URLs
        base_urls = [
            "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/",
            "https://github.com/multi30k/dataset/raw/master/data/task1/raw/",
            "https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/master/assets/data/"
        ]
        
        files_to_download = {
            'train.de': 'train.de',
            'train.en': 'train.en',
            'val.de': 'val.de',
            'val.en': 'val.en',
            'test_2016_flickr.de': 'test.de',
            'test_2016_flickr.en': 'test.en'
        }
        
        success = False
        for base_url in base_urls:
            try:
                for remote_file, local_file in files_to_download.items():
                    url = base_url + remote_file
                    output_path = os.path.join(self.data_dir, 'raw', local_file)
                    
                    if not os.path.exists(output_path):
                        logger.info(f"Downloading {remote_file} from {base_url}...")
                        urllib.request.urlretrieve(url, output_path)
                
                success = True
                logger.info(f"✅ Successfully downloaded from {base_url}")
                break
            except Exception as e:
                logger.warning(f"Failed to download from {base_url}: {e}")
                continue
        
        if not success:
            logger.error("❌ Failed to download from all sources")
            logger.info("Please install datasets library: pip install datasets")
            return [], [], []
        
        # Load data from files
        train_data = self._load_parallel_data('train')
        valid_data = self._load_parallel_data('val')
        test_data = self._load_parallel_data('test')
        
        return train_data, valid_data, test_data
    
    def _load_parallel_data(self, split: str) -> List[Tuple[str, str]]:
        """Load parallel data from files"""
        de_file = os.path.join(self.data_dir, 'raw', f'{split}.de')
        en_file = os.path.join(self.data_dir, 'raw', f'{split}.en')
        
        # Map val to valid for consistency
        if split == 'val':
            de_file = os.path.join(self.data_dir, 'raw', 'val.de')
            en_file = os.path.join(self.data_dir, 'raw', 'val.en')
        
        data = []
        
        try:
            with open(de_file, 'r', encoding='utf-8') as f_de, \
                 open(en_file, 'r', encoding='utf-8') as f_en:
                
                for de_line, en_line in zip(f_de, f_en):
                    de_line = de_line.strip()
                    en_line = en_line.strip()
                    if de_line and en_line:
                        data.append((de_line, en_line))
                        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return []
        
        return data
    
    def _save_data_to_files(self, train_data: List[Tuple[str, str]], 
                          valid_data: List[Tuple[str, str]], 
                          test_data: List[Tuple[str, str]]) -> None:
        """Save data to files for later use"""
        for data, split in [(train_data, 'train'), (valid_data, 'val'), (test_data, 'test')]:
            de_file = os.path.join(self.data_dir, 'raw', f'{split}.de')
            en_file = os.path.join(self.data_dir, 'raw', f'{split}.en')
            
            with open(de_file, 'w', encoding='utf-8') as f_de, \
                 open(en_file, 'w', encoding='utf-8') as f_en:
                for de_text, en_text in data:
                    f_de.write(de_text + '\n')
                    f_en.write(en_text + '\n')
    
    def train_sentencepiece(self, texts: List[str], model_prefix: str = "tokenizer", 
                          vocab_size: int = 10000) -> None:
        """Train SentencePiece tokenizer"""
        logger.info(f"Training SentencePiece model with vocab size {vocab_size}...")
        
        # Write texts to temporary file
        temp_file = os.path.join(self.data_dir, "temp_texts.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train model
        model_path = os.path.join(self.data_dir, 'processed', model_prefix)
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_path,
            vocab_size=vocab_size,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<bos>',
            eos_piece='<eos>',
            character_coverage=1.0  # Important for handling all characters
        )
        
        # Clean up
        os.remove(temp_file)
        logger.info(f"✅ SentencePiece model saved to {model_path}")
    
    def prepare_tokenizer(self, train_data: List[Tuple[str, str]]) -> None:
        """Prepare tokenizer from training data"""
        if os.path.exists(os.path.join(self.data_dir, 'processed', 'tokenizer.model')):
            logger.info("Tokenizer already exists. Skipping training.")
            return
        
        # Combine all texts for tokenizer training
        all_texts = []
        for src, tgt in train_data:
            all_texts.append(src)
            all_texts.append(tgt)
        
        # Train tokenizer
        self.train_sentencepiece(all_texts, "tokenizer", vocab_size=self.config['model']['vocab_size'])

if __name__ == "__main__":
    # Install datasets if not available
    if not HUGGINGFACE_AVAILABLE:
        import subprocess
        print("Installing datasets library...")
        subprocess.run(["pip", "install", "datasets", "-q"])
        from datasets import load_dataset
    
    downloader = DataDownloader()
    train_data, valid_data, test_data = downloader.download_multi30k()
    
    if train_data:
        # Train tokenizer
        downloader.prepare_tokenizer(train_data)
        logger.info("✅ Data download and tokenizer training completed!")
    else:
        logger.error("❌ Failed to download data.")