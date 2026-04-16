#!/usr/bin/env python3
"""Quick test of mlx-vlm Qwen2.5-VL model."""

import sys
from pathlib import Path

def test_mlx_vlm():
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
        model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        config = load_config("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model load failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Find a test image
    test_dirs = ["DRS", "DDS", "Error Codes", "Message Specs V2.1"]
    test_img = None
    for test_dir in test_dirs:
        img_dir = Path(test_dir) / "images"
        if img_dir.exists():
            pngs = list(img_dir.glob("*.png"))[:1]
            if pngs:
                test_img = pngs[0]
                break

    if not test_img:
        print("No test images found. Run convert_to_md.py first.")
        return

    print(f"✓ Test image: {test_img}")
    print()

    # Test alt text generation
    print("Testing image description ...")
    prompt_alt = "Describe this screenshot from a technical user guide in one concise sentence."
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
        print(f"✓ Alt text: {alt.strip()[:100]}...")
    except Exception as e:
        print(f"✗ Description failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()

    # Test text extraction
    print("Testing image text extraction ...")
    prompt_extract = "Extract all visible text from this image exactly as it appears."
    try:
        prompt_text = apply_chat_template(
            processor, config,
            prompt_extract,
            num_images=1
        )
        response = generate(
            model, processor,
            image=str(test_img),
            prompt=prompt_text,
            max_tokens=2048,
            verbose=False
        )
        text = response.text if hasattr(response, 'text') else str(response)
        extracted = text.strip()
        if extracted:
            print(f"✓ Extracted text ({len(extracted)} chars):")
            print(f"  {extracted[:200]}...")
        else:
            print("✓ No text extracted (image is likely non-text)")
    except Exception as e:
        print(f"✗ Extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("All tests passed. Model is working.")

if __name__ == "__main__":
    test_mlx_vlm()
