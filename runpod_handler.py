"""
RunPod Serverless Handler for OmniParser

Expected input format:
{
    "input": {
        "base64_image": "base64_encoded_image_string"
    }
}

Returns:
{
    "som_image_base64": "base64_encoded_annotated_image",
    "parsed_content_list": [...],
    "latency": 0.123
}
"""

import runpod
import os
import time
import traceback

from util.omniparser import Omniparser

# Global parser instance (loaded once per worker)
omniparser = None


def initialize_parser():
    """Initialize OmniParser with environment variables or defaults."""
    global omniparser

    if omniparser is not None:
        return omniparser

    print("Initializing OmniParser for RunPod serverless...")

    config = {
        'som_model_path': os.getenv('SOM_MODEL_PATH', 'weights/icon_detect/model.pt'),
        'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
        'caption_model_path': os.getenv('CAPTION_MODEL_PATH', 'weights/icon_caption_florence'),
        'BOX_TRESHOLD': float(os.getenv('BOX_TRESHOLD', '0.05')),
    }

    print(f"Config: {config}")

    if not os.path.exists(config['som_model_path']):
        raise RuntimeError(f"SOM model not found at {config['som_model_path']}")
    if not os.path.exists(config['caption_model_path']):
        raise RuntimeError(f"Caption model not found at {config['caption_model_path']}")

    omniparser = Omniparser(config)
    print("OmniParser initialized successfully")

    return omniparser


def handler(event):
    """RunPod handler function for serverless inference."""
    try:
        input_data = event.get("input", {})

        if not input_data:
            return {"error": "No input provided. Expected: {'input': {'base64_image': '...'}}"}

        base64_image = input_data.get("base64_image")

        if not base64_image:
            return {"error": "Missing 'base64_image' in input"}

        # Strip data URI prefix if present
        if isinstance(base64_image, str) and base64_image.startswith('data:'):
            base64_image = base64_image.split(',', 1)[1] if ',' in base64_image else base64_image
        base64_image = base64_image.strip()

        print(f"Processing image ({len(base64_image)} chars)")

        parser = initialize_parser()

        start = time.time()
        som_image_base64, parsed_content_list = parser.parse(base64_image)
        latency = time.time() - start

        print(f"Parsing completed in {latency:.3f}s")

        return {
            "som_image_base64": som_image_base64,
            "parsed_content_list": parsed_content_list,
            "latency": latency,
        }

    except Exception as e:
        error_msg = f"Error during parsing: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
