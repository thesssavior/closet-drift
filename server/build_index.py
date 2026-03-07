"""
Build Qdrant vector index from Fashionpedia dataset.

Usage:
  1. Download annotations:
     curl -L "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" -o data/fashionpedia_train.json
  2. Set env vars in .env.local: QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY
  3. Run: python build_index.py
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

COLLECTION = "fashionpedia"
EMBED_DIM = 3072  # gemini-embedding-001 output dim
BATCH_SIZE = 96  # Gemini free tier: 100 requests/min, batch to stay safe
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"

# Fashionpedia category mapping (27 apparel + 19 accessory categories)
FASHIONPEDIA_CATEGORIES = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater",
    3: "cardigan", 4: "jacket", 5: "vest", 6: "pants", 7: "shorts",
    8: "skirt", 9: "coat", 10: "dress", 11: "jumpsuit",
    12: "cape", 13: "glasses, sunglasses", 14: "hat", 15: "headband, head covering, hair accessory",
    16: "tie", 17: "glove", 18: "watch", 19: "belt",
    20: "leg warmer", 21: "tights, stockings", 22: "sock",
    23: "shoe", 24: "bag, wallet", 25: "scarf", 26: "umbrella",
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette",
    31: "sleeve", 32: "pocket", 33: "neckline", 34: "buckle",
    35: "zipper", 36: "applique", 37: "bead", 38: "bow",
    39: "flower", 40: "fringe", 41: "ribbon", 42: "rivet",
    43: "ruffle", 44: "sequin", 45: "tassel",
}

# Only index wearable garment/accessory categories (not parts like pocket, collar)
WEARABLE_IDS = set(range(27))

# Fashionpedia attribute mapping
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using Gemini embedding API."""
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
    """Create Qdrant collection (idempotent)."""
    # Check if exists
    r = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY},
        timeout=10,
    )
    if r.status_code == 200:
        print(f"Collection '{COLLECTION}' already exists. Deleting and recreating...")
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


def build_descriptions(annotations_path: str) -> list[dict]:
    """Parse Fashionpedia annotations into text descriptions."""
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}

    # Map image_id -> image info
    images = {img["id"]: img for img in data["images"]}

    items = []
    seen = set()

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in WEARABLE_IDS:
            continue

        cat_name = categories.get(cat_id, FASHIONPEDIA_CATEGORIES.get(cat_id, "clothing"))
        attr_ids = ann.get("attribute_ids", [])
        attr_names = [attributes.get(a, FASHIONPEDIA_ATTRIBUTES.get(a, "")) for a in attr_ids]
        attr_names = [a for a in attr_names if a]

        # Build description
        desc_parts = []
        if attr_names:
            desc_parts.append(", ".join(attr_names))
        desc_parts.append(cat_name)
        description = " ".join(desc_parts)

        # Deduplicate similar descriptions
        dedup_key = description.lower().strip()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        img_info = images.get(ann["image_id"], {})
        img_filename = img_info.get("file_name", "")

        items.append({
            "id": len(items),
            "description": description,
            "category": cat_name,
            "category_id": cat_id,
            "attributes": attr_names,
            "image_url": f"https://s3.amazonaws.com/ifashionist-dataset/images/train2020/{img_filename}" if img_filename else "",
            "image_filename": img_filename,
        })

    print(f"Built {len(items)} unique item descriptions")
    return items


def upsert_batch(points: list[dict]):
    """Upsert a batch of points to Qdrant."""
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"points": points},
        timeout=30,
    )
    resp.raise_for_status()


def main():
    annotations_path = Path(__file__).parent / "data" / "fashionpedia_train.json"
    if not annotations_path.exists():
        print("ERROR: Download annotations first:")
        print('  curl -L "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json" -o data/fashionpedia_train.json')
        return

    items = build_descriptions(str(annotations_path))
    create_collection()

    total = len(items)
    for i in range(0, total, BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
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
                    "image_url": item["image_url"],
                },
            })

        upsert_batch(points)
        print(f"  [{i + len(batch)}/{total}] upserted")

        # Rate limit: Gemini free tier is 100 req/min, we batch so ~1 req per batch
        if i + BATCH_SIZE < total:
            time.sleep(1)

    print(f"\nDone! {total} items indexed in '{COLLECTION}'")


if __name__ == "__main__":
    main()
