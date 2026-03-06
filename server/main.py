import io
import os
import base64
import hashlib
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Load .env.local from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env.local")
load_dotenv(_project_root / ".env")

import numpy as np
import requests as http_requests
import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# Friendly search-friendly names
SEARCH_NAMES = {
    1: "hat",
    3: "sunglasses",
    4: "top shirt jacket",
    5: "skirt",
    6: "pants trousers",
    7: "dress",
    8: "belt",
    9: "shoes",
    10: "shoes",
    16: "bag",
    17: "scarf",
}

CLOTHING_IDS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17}

processor = None
model = None

# Cache: image_hash -> { seg_map, orig_w, orig_h, image_bytes }
seg_cache: dict[str, dict] = {}


def get_model():
    global processor, model
    if processor is None:
        print("Loading SegFormer model...")
        processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        model.eval()
        print("Model loaded.")
    return processor, model


def image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def dominant_color_name(img: Image.Image, mask: np.ndarray) -> str:
    """Get the dominant color name from masked region of image."""
    arr = np.array(img)
    pixels = arr[mask > 0]
    if len(pixels) == 0:
        return ""

    # Subsample for speed
    if len(pixels) > 1000:
        idx = np.random.choice(len(pixels), 1000, replace=False)
        pixels = pixels[idx]

    avg = pixels.mean(axis=0).astype(int)
    r, g, b = int(avg[0]), int(avg[1]), int(avg[2])

    # Map to basic color names
    hsv_img = Image.new("RGB", (1, 1), (r, g, b)).convert("HSV")
    h, s, v = hsv_img.getpixel((0, 0))

    if v < 40:
        return "black"
    if s < 25 and v > 200:
        return "white"
    if s < 30:
        return "gray"

    # Hue-based (0-255 scale in PIL)
    hue = h * 2  # convert to 0-360
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 160:
        return "green"
    if hue < 250:
        return "blue"
    if hue < 300:
        return "purple"
    return "pink"


def search_products(query: str) -> list[dict]:
    """Search Google Shopping via Serper.dev API."""
    if not SERPER_API_KEY:
        return []

    try:
        resp = http_requests.post(
            "https://google.serper.dev/shopping",
            json={"q": query, "num": 8},
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("shopping", [])[:8]:
            results.append({
                "id": item.get("position", 0),
                "name": item.get("title", ""),
                "price": item.get("price", ""),
                "image": item.get("imageUrl", ""),
                "brand": item.get("source", ""),
                "link": item.get("link", ""),
            })
        return results
    except Exception as e:
        print(f"[search] error: {e}")
        return []


@app.on_event("startup")
async def startup():
    get_model()


@app.post("/api/encode")
async def encode_image(file: UploadFile = File(...)):
    proc, mdl = get_model()
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    h = image_hash(data)

    if h not in seg_cache:
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = mdl(**inputs)

        logits = outputs.logits
        upsampled = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        seg_cache[h] = {
            "seg_map": seg_map,
            "orig_w": img.size[0],
            "orig_h": img.size[1],
            "image": img,
        }

        unique = np.unique(seg_map)
        detected = [LABELS[int(u)] for u in unique if int(u) in CLOTHING_IDS]
        print(f"[encode] {img.size[0]}x{img.size[1]} detected: {detected}")

    entry = seg_cache[h]
    return {"hash": h, "width": entry["orig_w"], "height": entry["orig_h"]}


@app.post("/api/decode")
async def decode_mask(body: dict):
    h = body["hash"]
    x = float(body["x"])
    y = float(body["y"])
    canvas_w = int(body["canvasWidth"])
    canvas_h = int(body["canvasHeight"])

    if h not in seg_cache:
        return JSONResponse({"error": "Image not encoded"}, 400)

    entry = seg_cache[h]
    seg_map = entry["seg_map"]
    orig_w = entry["orig_w"]
    orig_h = entry["orig_h"]

    max_dim = max(orig_w, orig_h)
    model_scale = 1024 / max_dim
    orig_x = max(0, min(int(x / model_scale), orig_w - 1))
    orig_y = max(0, min(int(y / model_scale), orig_h - 1))

    category_id = int(seg_map[orig_y, orig_x])
    category_name = LABELS.get(category_id, "Unknown")

    if category_id not in CLOTHING_IDS:
        return {"mask": None, "category": category_name, "categoryId": category_id}

    # Build mask and smooth edges
    mask = (seg_map == category_id).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize((canvas_w, canvas_h), Image.BILINEAR)
    # Smooth edges with a slight blur
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # Bounding box
    mask_arr = np.array(mask_img)
    ys, xs = np.where(mask_arr > 0)
    bbox = None
    if len(xs) > 0:
        bbox = {
            "x": int(xs.min()),
            "y": int(ys.min()),
            "w": int(xs.max() - xs.min()),
            "h": int(ys.max() - ys.min()),
        }

    return {
        "mask": mask_b64,
        "category": category_name,
        "categoryId": category_id,
        "bbox": bbox,
    }


@app.post("/api/search")
async def search_item(body: dict):
    """Search for similar products based on clicked clothing item."""
    h = body["hash"]
    category_id = int(body["categoryId"])

    if h not in seg_cache:
        return JSONResponse({"error": "Image not encoded"}, 400)

    entry = seg_cache[h]
    seg_map = entry["seg_map"]
    img = entry["image"]

    mask = (seg_map == category_id).astype(np.uint8) * 255
    color = dominant_color_name(img, mask)
    search_term = SEARCH_NAMES.get(category_id, LABELS.get(category_id, "clothing"))
    query = f"{color} {search_term}".strip()

    print(f"[search] category={LABELS[category_id]} color={color} query=\"{query}\"")

    products = search_products(query)

    return {
        "query": query,
        "products": products,
    }


@app.get("/api/health")
async def health():
    has_key = bool(SERPER_API_KEY)
    return {"status": "ok", "searchEnabled": has_key}
