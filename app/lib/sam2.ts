export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface EncodeResult {
  hash: string;
  width: number;
  height: number;
}

export interface DecodeResult {
  mask: string | null;
  category: string;
  categoryId: number;
  bbox: { x: number; y: number; w: number; h: number } | null;
}

export interface Product {
  id: number;
  name: string;
  price: string;
  image: string;
  brand: string;
  link: string;
}

export interface SearchResult {
  query: string;
  products: Product[];
}

export async function encodeImage(file: File): Promise<EncodeResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/encode`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Encode failed: ${res.statusText}`);
  return res.json();
}

export async function decodeMask(
  hash: string,
  x: number,
  y: number,
  canvasWidth: number,
  canvasHeight: number
): Promise<DecodeResult> {
  const res = await fetch(`${API_BASE}/api/decode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, x, y, canvasWidth, canvasHeight }),
  });
  if (!res.ok) throw new Error(`Decode failed: ${res.statusText}`);
  return res.json();
}

export async function fetchClothingMask(
  hash: string,
  canvasWidth: number,
  canvasHeight: number
): Promise<{ mask: string }> {
  const res = await fetch(`${API_BASE}/api/clothing-mask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, canvasWidth, canvasHeight }),
  });
  if (!res.ok) throw new Error(`Clothing mask failed: ${res.statusText}`);
  return res.json();
}

export async function searchProducts(
  hash: string,
  categoryId: number
): Promise<SearchResult> {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash, categoryId }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
  return res.json();
}
