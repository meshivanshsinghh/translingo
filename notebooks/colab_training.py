#!/usr/bin/env python3
"""
TransLingo Training Script for Google Colab
This script is designed to be copied and run in Google Colab cells
"""

# ============================================
# CELL 1: Check GPU and Setup
# ============================================
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# CELL 2: Mount Google Drive (Optional)
# ============================================
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_PATH = '/content/drive/MyDrive/translingo_checkpoints'
# import os
# os.makedirs(DRIVE_PATH, exist_ok=True)

# ============================================
# CELL 3: Clone Repository
# ============================================
# !git clone https://github.com/YOUR_USERNAME/translingo.git
# %cd translingo
# !pip install -r requirements.txt

# ============================================
# CELL 4: Main Training Script
# ============================================
import os
import sys
import yaml
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download import DataDownloader
from data.preprocessing import create_dataloaders
from training.train import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function for Colab"""
    
    # 1. Download and prepare data
    logger.info("Downloading Multi30k dataset...")
    downloader = DataDownloader()
    train_data, valid_data, test_data = downloader.download_multi30k()
    
    if not train_data:
        logger.error("Failed to download data!")
        return
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Valid samples: {len(valid_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # 2. Train tokenizer
    logger.info("Training tokenizer...")
    downloader.prepare_tokenizer(train_data)
    
    # 3. Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer('configs/config.yaml')
    
    # Optional: Modify config for faster testing
    # trainer.config['training']['num_epochs'] = 5
    # trainer.config['model']['n_layers'] = 2  # Fewer layers for testing
    
    # 4. Create dataloaders
    tokenizer_path = os.path.join('data', 'processed', 'tokenizer.model')
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_data,
        valid_data,
        test_data,
        tokenizer_path,
        batch_size=trainer.config['training']['batch_size'],
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Valid batches: {len(valid_loader)}")
    
    # 5. Train model
    logger.info("Starting training...")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Epochs: {trainer.config['training']['num_epochs']}")
    logger.info(f"Batch size: {trainer.config['training']['batch_size']}")
    
    trainer.train(train_loader, valid_loader)
    
    # 6. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test BLEU: {test_metrics['bleu']:.2f}")
    
    # 7. Save to Google Drive if mounted
    if 'DRIVE_PATH' in globals():
        os.system(f"cp -r checkpoints/* {DRIVE_PATH}/")
        os.system(f"cp -r logs/* {DRIVE_PATH}/logs/")
        logger.info(f"Results saved to Google Drive: {DRIVE_PATH}")

if __name__ == "__main__":
    main()

# ============================================
# CELL 5: Quick Translation Test
# ============================================
"""
import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load('data/processed/tokenizer.model')

# Test sentences
test_sentences = [
    "Guten Morgen!",
    "Wie geht es dir?",
    "Das Wetter ist heute schön."
]

# Load best model
checkpoint = torch.load('checkpoints/best.pt', map_location='cuda')
model = create_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to('cuda')

# Translate
with torch.no_grad():
    for sentence in test_sentences:
        tokens = sp.encode(sentence)
        src = torch.tensor([sp.bos_id()] + tokens + [sp.eos_id()]).unsqueeze(0).cuda()
        translation = model.generate(src, max_length=50)
        translated_tokens = translation[0].cpu().numpy().tolist()
        translated_text = sp.decode(translated_tokens)
        print(f"Source: {sentence}")
        print(f"Translation: {translated_text}")
        print()
"""

# ============================================
# CELL 6: Download Results
# ============================================
"""
# Zip and download checkpoints
!zip -r translingo_checkpoints.zip checkpoints/
!zip -r translingo_logs.zip logs/

from google.colab import files
files.download('translingo_checkpoints.zip')
files.download('translingo_logs.zip')
"""
