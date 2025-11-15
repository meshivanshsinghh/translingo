# app.py
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.gradio_app import TransLingoDemo

# Create demo with checkpoint path
demo_app = TransLingoDemo(checkpoint_path="data/best.pt")

# Create interface
demo = demo_app.create_interface()

# Launch with HuggingFace Spaces settings
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # HF Spaces handles sharing
    )