#!/usr/bin/env python3
"""Download all model weights required by OmniParser.

Usage:
    python download_models.py

Downloads:
    1. icon_detect (YOLO) — fine-tuned icon detection model from HuggingFace
    2. icon_caption_florence — fine-tuned Florence-2 caption model from HuggingFace
    3. Florence-2-base processor — tokenizer/processor from microsoft/Florence-2-base
    4. PaddleOCR — detection + recognition models (auto-downloaded by paddleocr)
    5. EasyOCR — English text recognition model (auto-downloaded by easyocr)
"""
import os
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


def download_huggingface_models():
    """Download fine-tuned icon_detect and icon_caption from HuggingFace."""
    from huggingface_hub import snapshot_download

    print("[1/5] Downloading icon_detect (YOLO) weights...")
    snapshot_download(
        'microsoft/OmniParser-v2.0',
        local_dir=WEIGHTS_DIR,
        allow_patterns=['icon_detect/*'],
    )
    detect_path = os.path.join(WEIGHTS_DIR, 'icon_detect', 'model.pt')
    assert os.path.exists(detect_path), f"Expected {detect_path}"
    size_mb = os.path.getsize(detect_path) / 1e6
    print(f"       -> icon_detect/model.pt ({size_mb:.0f}MB)")

    print("[2/5] Downloading icon_caption_florence weights...")
    snapshot_download(
        'microsoft/OmniParser-v2.0',
        local_dir=WEIGHTS_DIR,
        allow_patterns=['icon_caption/*'],
    )
    # Rename to match expected path
    src = os.path.join(WEIGHTS_DIR, 'icon_caption')
    dst = os.path.join(WEIGHTS_DIR, 'icon_caption_florence')
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print("       -> renamed icon_caption/ to icon_caption_florence/")
    elif os.path.exists(dst):
        print("       -> icon_caption_florence/ already exists")
    safetensors = os.path.join(dst, 'model.safetensors')
    if os.path.exists(safetensors):
        size_mb = os.path.getsize(safetensors) / 1e6
        print(f"       -> icon_caption_florence/model.safetensors ({size_mb:.0f}MB)")


def download_florence2_processor():
    """Download the Florence-2-base processor (tokenizer + image processor)."""
    print("[3/5] Downloading Florence-2-base processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    print(f"       -> cached ({type(processor).__name__})")


def download_paddleocr():
    """Trigger PaddleOCR model download by initializing it."""
    print("[4/5] Downloading PaddleOCR models...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Run a dummy inference to ensure all sub-models are downloaded
    import numpy as np
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy[20:80, 20:80] = 255
    ocr.ocr(dummy, cls=False)
    print("       -> PaddleOCR models cached")


def download_easyocr():
    """Trigger EasyOCR model download by initializing it."""
    print("[5/5] Downloading EasyOCR models...")
    import easyocr
    reader = easyocr.Reader(['en'], verbose=False)
    print("       -> EasyOCR English models cached")


def main():
    print(f"OmniParser Model Setup")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print("=" * 50)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    download_huggingface_models()
    download_florence2_processor()
    download_paddleocr()
    download_easyocr()

    print("=" * 50)
    print("All models downloaded successfully.")
    print(f"\nWeights directory contents:")
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        level = root.replace(WEIGHTS_DIR, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            if size > 1e6:
                print(f"{indent}  {f} ({size/1e6:.0f}MB)")
            else:
                print(f"{indent}  {f} ({size/1e3:.0f}KB)")


if __name__ == '__main__':
    main()
