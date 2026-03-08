"""
Build Qdrant vector index from Fashionpedia val dataset with locally cropped garment images.

Usage:
  1. Ensure fashionpedia/annotations/instances_attributes_val2020.json exists
  2. Ensure fashionpedia/images/val_test/test/ contains the images
  3. Set env vars in .env.local: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
  4. Run: python build_index_local.py
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

COLLECTION = "fashionpedia"
EMBED_DIM = 3072
BATCH_SIZE = 96
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"

WEARABLE_IDS = set(range(27))

FASHIONPEDIA_ATTRIBUTES = {
    0: "floral", 1: "graphic", 2: "striped", 3: "plain", 4: "lattice",
    5: "spotted", 6: "checked", 7: "solid color",
    8: "denim", 9: "chiffon", 10: "corduroy", 11: "cotton",
    12: "faux fur", 13: "knit", 14: "lace", 15: "leather",
    16: "linen", 17: "mesh", 18: "nylon", 19: "satin",
    20: "sequined", 21: "silk", 22: "suede", 23: "velvet", 24: "wool",
    25: "long sleeve", 26: "short sleeve", 27: "sleeveless",
    28: "maxi length", 29: "midi length", 30: "mini length",
    31: "crew neckline", 32: "v-neckline", 33: "turtleneck",
    34: "sweetheart neckline", 35: "straight fit", 36: "loose fit", 37: "tight fit",
}

DATA_DIR = Path(__file__).parent / "fashionpedia"
IMAGES_DIR = DATA_DIR / "images" / "val_test" / "test"
CROPS_DIR = Path(__file__).parent / "static" / "crops"


def embed_texts(texts: list[str]) -> list[list[float]]:
    body = {
        "requests": [
            {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    resp = requests.post(GEMINI_EMBED_URL, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [e["values"] for e in data["embeddings"]]


def create_collection():
    r = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY},
        timeout=10,
    )
    if r.status_code == 200:
        print(f"Collection '{COLLECTION}' exists. Deleting and recreating...")
        requests.delete(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            headers={"api-key": QDRANT_API_KEY},
            timeout=10,
        )

    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 0},
        },
        timeout=10,
    )
    resp.raise_for_status()
    print(f"Created collection '{COLLECTION}'")


def crop_garment(image_path: Path, bbox: list, ann_id: int, padding: float = 0.1) -> str | None:
    """Crop garment from image using COCO bbox [x, y, w, h] with padding. Returns crop filename."""
    crop_filename = f"{ann_id}.jpg"
    crop_path = CROPS_DIR / crop_filename
    if crop_path.exists():
        return crop_filename

    try:
        img = Image.open(image_path)
    except Exception:
        return None

    x, y, w, h = bbox
    img_w, img_h = img.size

    # Add padding
    pad_x = w * padding
    pad_y = h * padding
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(img_w, int(x + w + pad_x))
    bottom = min(img_h, int(y + h + pad_y))

    if right - left < 10 or bottom - top < 10:
        return None

    crop = img.crop((left, top, right, bottom))
    # Resize to max 400px on longest side for storage efficiency
    max_side = max(crop.size)
    if max_side > 400:
        scale = 400 / max_side
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)

    crop.save(crop_path, "JPEG", quality=85)
    return crop_filename


def build_items(annotations_path: str) -> list[dict]:
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}
    images = {img["id"]: img for img in data["images"]}

    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    skipped = 0

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in WEARABLE_IDS:
            continue

        img_info = images.get(ann["image_id"], {})
        img_filename = img_info.get("file_name", "")
        img_path = IMAGES_DIR / img_filename

        if not img_path.exists():
            skipped += 1
            continue

        bbox = ann.get("bbox")
        if not bbox:
            skipped += 1
            continue

        crop_filename = crop_garment(img_path, bbox, ann["id"])
        if not crop_filename:
            skipped += 1
            continue

        cat_name = categories.get(cat_id, "clothing")
        attr_ids = ann.get("attribute_ids", [])
        attr_names = [attributes.get(a, FASHIONPEDIA_ATTRIBUTES.get(a, "")) for a in attr_ids]
        attr_names = [a for a in attr_names if a]

        desc_parts = []
        if attr_names:
            desc_parts.append(", ".join(attr_names))
        desc_parts.append(cat_name)
        description = " ".join(desc_parts)

        items.append({
            "id": len(items),
            "description": description,
            "category": cat_name,
            "category_id": cat_id,
            "attributes": attr_names,
            "crop_filename": crop_filename,
        })

    print(f"Built {len(items)} items ({skipped} skipped)")
    return items


def upsert_batch(points: list[dict]):
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"points": points},
        timeout=30,
    )
    resp.raise_for_status()


def main():
    annotations_path = DATA_DIR / "annotations" / "instances_attributes_val2020.json"
    if not annotations_path.exists():
        print("ERROR: annotations not found")
        return

    items = build_items(str(annotations_path))
    if not items:
        print("No items to index")
        return

    create_collection()

    total = len(items)
    for i in range(0, total, BATCH_SIZE):
        batch = items[i: i + BATCH_SIZE]
        texts = [item["description"] for item in batch]

        try:
            vectors = embed_texts(texts)
        except Exception as e:
            print(f"Embed error at batch {i}: {e}")
            print("Waiting 60s for rate limit...")
            time.sleep(60)
            vectors = embed_texts(texts)

        points = []
        for item, vector in zip(batch, vectors):
            points.append({
                "id": item["id"],
                "vector": vector,
                "payload": {
                    "description": item["description"],
                    "category": item["category"],
                    "category_id": item["category_id"],
                    "attributes": item["attributes"],
                    "crop_filename": item["crop_filename"],
                },
            })

        upsert_batch(points)
        print(f"  [{i + len(batch)}/{total}] upserted")

        if i + BATCH_SIZE < total:
            time.sleep(1)

    print(f"\nDone! {total} items indexed in '{COLLECTION}'")


if __name__ == "__main__":
    main()
