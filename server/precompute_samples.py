"""Pre-compute segmentation maps for sample images.

Run once at build/deploy time:
    python precompute_samples.py

Outputs to static/sample_cache/:
    manifest.json  — { filename: { hash, orig_w, orig_h } }
    {hash}.npy     — seg_map array (uint8)
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from transformers import SegformerImageProcessor
from optimum.onnxruntime import ORTModelForSemanticSegmentation

SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).parent.parent / "public" / "samples"))
CACHE_DIR = Path(__file__).parent / "static" / "sample_cache"


def main():
    print("Loading SegFormer model...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = ORTModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes", export=True
    )
    print("Model loaded.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        print(f"Processing {path.name}...")
        data = path.read_bytes()
        h = hashlib.md5(data).hexdigest()
        img = Image.open(path).convert("RGB")

        inputs = processor(images=img, return_tensors="np")
        outputs = model(**{k: v for k, v in inputs.items()})

        logits = np.array(outputs.logits)
        target_h, target_w = img.size[1], img.size[0]
        _, _, lh, lw = logits.shape
        upsampled = zoom(logits, (1, 1, target_h / lh, target_w / lw), order=1)
        seg_map = upsampled.argmax(axis=1)[0].astype(np.uint8)

        np.save(CACHE_DIR / f"{h}.npy", seg_map)
        shutil.copy2(path, CACHE_DIR / f"{h}{path.suffix.lower()}")
        manifest[path.name] = {"hash": h, "orig_w": img.size[0], "orig_h": img.size[1], "ext": path.suffix.lower()}
        print(f"  hash={h}  size={img.size[0]}x{img.size[1]}")

    with open(CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} samples cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()
