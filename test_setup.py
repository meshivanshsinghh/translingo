#!/usr/bin/env python3
"""
Test script to verify TransLingo setup
"""

import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    logger.info("Testing imports...")
    
    modules = [
        "data.download",
        "data.preprocessing",
        "model.transformer",
        "model.attention",
        "model.embeddings",
        "model.layers",
        "training.train",
        "training.loss",
        "training.optimizer",
        "inference.beam_search",
        "inference.translate",
        "utils.metrics",
        "frontend.gradio_app"
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except Exception as e:
            logger.error(f"❌ {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_cuda():
    """Test CUDA availability"""
    logger.info("\nTesting CUDA...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available - training will be slow!")
    
    return True

def test_model_creation():
    """Test if model can be created"""
    logger.info("\nTesting model creation...")
    
    try:
        from model.transformer import Transformer
        
        # Create small test model
        model = Transformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_length=50
        )
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        src = torch.randint(0, 1000, (2, 10)).to(device)
        tgt = torch.randint(0, 1000, (2, 10)).to(device)
        
        output = model(src, tgt)
        logger.info(f"✅ Model output shape: {output.shape}")
        logger.info(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return False

def test_data_directory():
    """Test data directory structure"""
    logger.info("\nChecking directory structure...")
    
    dirs_to_check = [
        "data",
        "data/raw",
        "data/processed",
        "model",
        "training",
        "inference",
        "utils",
        "api",
        "frontend",
        "notebooks",
        "configs"
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            logger.info(f"✅ {dir_path}")
        else:
            logger.error(f"❌ {dir_path} - missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("TransLingo Setup Test")
    logger.info("=" * 50)
    
    tests = [
        ("Directory Structure", test_data_directory),
        ("Module Imports", test_imports),
        ("CUDA/GPU", test_cuda),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! You're ready to start training.")
        logger.info("\nNext steps:")
        logger.info("1. Upload notebooks/colab_training.py to Google Colab")
        logger.info("2. Run training on Colab with GPU")
        logger.info("3. Download checkpoints when training completes")
        logger.info("4. Run: python frontend/gradio_app.py")
    else:
        logger.error("\n❌ Some tests failed. Please fix the issues before proceeding.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
