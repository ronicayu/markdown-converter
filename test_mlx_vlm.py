#!/usr/bin/env python3
"""Quick test of mlx-vlm Qwen2.5-VL model."""

import sys
import argparse
from pathlib import Path

def test_mlx_vlm(image_path: str = None):
    """Load model and test on a sample image."""
    print("Loading mlx-vlm Qwen2.5-VL-7B-Instruct-4bit ...")
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model_id = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        model, processor = load(model_id)
        config = load_config(model_id)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model load failed: {e}", file=sys.stderr)
        sys.exit(1)

    test_img = None
    if image_path:
        test_img = Path(image_path)
    else:
        # Try to find any PNG in current directory or out/images/
        search_paths = [Path("."), Path("out/images")]
        for p in search_paths:
            if p.exists():
                pngs = list(p.rglob("*.png")) + list(p.rglob("*.jpg"))
                if pngs:
                    test_img = pngs[0]
                    break

    if not test_img or not test_img.exists():
        print("No test image found. Please provide one with --image path/to/img.png")
        return

    print(f"✓ Test image: {test_img}")
    print()

    print("Testing image description ...")
    prompt_alt = "Describe this image in one concise sentence."
    try:
        prompt_text = apply_chat_template(
            processor, config,
            prompt_alt,
            num_images=1
        )
        response = generate(
            model, processor,
            image=str(test_img),
            prompt=prompt_text,
            max_tokens=200,
            verbose=False
        )
        alt = response.text if hasattr(response, 'text') else str(response)
        print(f"✓ Description: {alt.strip()}")
    except Exception as e:
        print(f"✗ Description failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("All tests passed. Model is working.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mlx-vlm model")
    parser.add_argument("--image", help="Path to a test image")
    args = parser.parse_args()
    test_mlx_vlm(args.image)
