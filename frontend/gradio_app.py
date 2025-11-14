import gradio as gr
import torch
import os
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from model.transformer import Transformer
from inference.translate import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransLingoDemo:
    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize TransLingo demo"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to find checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        
        self.model_loaded = False
        self.translator = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self._load_model(checkpoint_path)
                self.model_loaded = True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint automatically"""
        checkpoint_dir = "checkpoints"
        
        # Look for best.pt first
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            return best_path
        
        # Then look for latest.pt
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        # Find any checkpoint
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                return os.path.join(checkpoint_dir, checkpoints[0])
        
        return None
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        self.model = Transformer(
            vocab_size=config['model']['vocab_size'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            d_ff=config['model']['d_ff'],
            max_seq_length=config['model']['max_seq_length'],
            dropout=0.0
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create translator
        tokenizer_path = os.path.join('data', 'processed', 'tokenizer.model')
        self.translator = Translator(
            self.model,
            tokenizer_path,
            self.device,
            beam_size=4,
            use_beam_search=True
        )
        
        # Store metadata
        self.model_info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'bleu': checkpoint.get('bleu', 0.0),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Epoch: {self.model_info['epoch']}, BLEU: {self.model_info['bleu']:.2f}")
    
    def translate(self, text: str, use_beam_search: bool = True) -> str:
        """Translate text"""
        if not self.model_loaded:
            return "❌ No model loaded. Please train the model first using Google Colab."
        
        if not text.strip():
            return ""
        
        try:
            # Update decoder settings
            self.translator.use_beam_search = use_beam_search
            
            # Translate
            translation = self.translator.translate(text.strip())
            return translation
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"❌ Translation error: {str(e)}"
    
    def get_model_info(self) -> str:
        """Get model information"""
        if not self.model_loaded:
            return "No model loaded"
        
        info = f"""
        **Model Information:**
        - Device: {self.device}
        - Parameters: {self.model_info['parameters']:,}
        - Training Epoch: {self.model_info['epoch']}
        - BLEU Score: {self.model_info['bleu']:.2f}
        - Architecture: {self.model.n_layers} layers, {self.model.d_model} dim, {self.model.n_heads} heads
        """
        return info.strip()
    
    def create_interface(self):
        """Create Gradio interface"""
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .output-text {
            font-size: 16px;
            line-height: 1.6;
        }
        """
        
        with gr.Blocks(title="TransLingo - Neural Machine Translation", css=css) as interface:
            # Header
            gr.Markdown(
                """
                # 🌐 TransLingo - Neural Machine Translation
                
                Translate German text to English using a custom-built Transformer model.
                
                ---
                """
            )
            
            # Model info
            with gr.Row():
                with gr.Column(scale=1):
                    model_info = gr.Markdown(self.get_model_info())
            
            # Translation interface
            with gr.Row():
                with gr.Column(scale=1):
                    # Input
                    input_text = gr.Textbox(
                        label="German Text",
                        placeholder="Geben Sie hier deutschen Text ein...",
                        lines=5,
                        max_lines=10
                    )
                    
                    # Options
                    with gr.Row():
                        beam_search = gr.Checkbox(
                            label="Use Beam Search",
                            value=True,
                            info="Better quality but slower"
                        )
                        translate_btn = gr.Button("Translate 🚀", variant="primary")
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            "Guten Morgen! Wie geht es dir heute?",
                            "Das Wetter ist heute sehr schön.",
                            "Ich liebe es, neue Sprachen zu lernen.",
                            "Der Hund spielt im Garten mit dem Ball.",
                            "Können Sie mir bitte helfen?",
                            "Die Katze sitzt auf dem Dach.",
                            "Ich möchte ein Glas Wasser, bitte.",
                            "Wir gehen morgen ins Kino."
                        ],
                        inputs=input_text,
                        label="Example Sentences"
                    )
                
                with gr.Column(scale=1):
                    # Output
                    output_text = gr.Textbox(
                        label="English Translation",
                        lines=5,
                        max_lines=10,
                        interactive=False,
                        elem_classes=["output-text"]
                    )
                    
                    # Additional features
                    with gr.Accordion("Translation Options", open=False):
                        gr.Markdown(
                            """
                            - **Beam Search**: Uses beam search decoding for better quality
                            - **Greedy**: Faster but potentially lower quality
                            
                            The model was trained on the Multi30k dataset (German-English).
                            """
                        )
            
            # Translation history
            with gr.Accordion("Recent Translations", open=False):
                history = gr.Dataframe(
                    headers=["German", "English", "Method"],
                    datatype=["str", "str", "str"],
                    row_count=5,
                    col_count=(3, "fixed"),
                    interactive=False
                )
            
            # Connect events
            translate_btn.click(
                fn=self.translate,
                inputs=[input_text, beam_search],
                outputs=output_text
            )
            
            input_text.submit(
                fn=self.translate,
                inputs=[input_text, beam_search],
                outputs=output_text
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                
                ### About TransLingo
                
                This is a neural machine translation system built from scratch using PyTorch.
                The model uses the Transformer architecture with:
                - Multi-head attention mechanism
                - Positional encoding
                - Label smoothing
                - Beam search decoding
                
                **Note**: This is a demonstration model trained on limited data.
                For production use, consider training on larger datasets.
                """
            )
        
        return interface
    
    def launch(self, share: bool = True, server_port: int = 7860):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on port {server_port}")
        if share:
            logger.info("Creating public URL...")
        
        interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            show_error=True
        )


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TransLingo Gradio Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Don't create public URL"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = TransLingoDemo(checkpoint_path=args.checkpoint)
    demo.launch(share=not args.no_share, server_port=args.port)


if __name__ == "__main__":
    main()
