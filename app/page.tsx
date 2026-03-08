"use client";

import { useState, useRef, useCallback, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { MasonryPhotoAlbum } from "react-photo-album";
import "react-photo-album/masonry.css";
import {
  encodeImage,
  decodeMask,
  searchProducts,
  findSimilar,
  fetchClothingMask,
  fetchSampleHashes,
  API_BASE,
  type Product,
} from "./lib/sam2";

type Stage = "idle" | "encoding" | "ready";

const SAMPLE_PHOTOS = [
  { src: "/samples/1.jpeg", width: 736, height: 1104 },
  { src: "/samples/6.jpg", width: 1920, height: 1280 },
  { src: "/samples/2.jpg", width: 1707, height: 2560 },
  { src: "/samples/4.jpg", width: 1110, height: 1665 },
  { src: "/samples/5.jpg", width: 1024, height: 683 },
  { src: "/samples/3.jpg", width: 540, height: 360 },
];

function computeBboxFromMask(maskData: ImageData, padding = 16) {
  const { width, height, data } = maskData;
  let minX = width, minY = height, maxX = 0, maxY = 0;
  let found = false;
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      if (data[(py * width + px) * 4] > 20) {
        found = true;
        minX = Math.min(minX, px);
        maxX = Math.max(maxX, px);
        minY = Math.min(minY, py);
        maxY = Math.max(maxY, py);
      }
    }
  }
  if (!found) return null;
  const x = Math.max(0, minX - padding);
  const y = Math.max(0, minY - padding);
  const w = Math.min(width - x, maxX - minX + 1 + padding * 2);
  const h = Math.min(height - y, maxY - minY + 1 + padding * 2);
  return { x, y, w, h };
}

export default function Page() {
  return (
    <Suspense>
      <Home />
    </Suspense>
  );
}

function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [stage, setStage] = useState<Stage>("idle");
  const [statusText, setStatusText] = useState("");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [landingKey, setLandingKey] = useState(0);
  const [products, setProducts] = useState<Product[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [focusBox, setFocusBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [focusDimmed, setFocusDimmed] = useState(false);
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const offsetRef = useRef(0);
  // Track current similar-search point ID for infinite scroll
  const similarIdRef = useRef<number | null>(null);
  // Track current text-search params for infinite scroll
  const textSearchRef = useRef<{ hash: string; categoryId: number } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayARef = useRef<HTMLCanvasElement>(null);
  const overlayBRef = useRef<HTMLCanvasElement>(null);
  const activeOverlayRef = useRef<"A" | "B">("A");
  const imageRef = useRef<HTMLImageElement | null>(null);
  const hashRef = useRef("");
  const fileRef = useRef<File | null>(null);
  const dimsRef = useRef({ scale: 1, origW: 0, origH: 0 });
  const hoverThrottleRef = useRef(0);
  const lastCategoryRef = useRef(-1);
  const lastHoverResultRef = useRef<{ mask: string; category: string; categoryId: number } | null>(null);
  const maskImageDataRef = useRef<ImageData | null>(null);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const focusDimmedRef = useRef(false);
  const clickGenRef = useRef(0);
  const startDripRef = useRef<((maskB64: string) => void) | null>(null);
  const dripMaskUrlRef = useRef<string | null>(null);
  const fullMaskB64Ref = useRef<string | null>(null);
  const focusMaskUrlRef = useRef<string | null>(null);
  const [dripState, setDripState] = useState<{ maskUrl: string; width: number; height: number } | null>(null);
  const [dripVisible, setDripVisible] = useState(false);
  const [focusMaskUrl, setFocusMaskUrl] = useState<string | null>(null);
  const dripId = useRef(`d${Math.random().toString(36).slice(2, 8)}`).current;
  const sampleHashesRef = useRef<Record<string, string>>({});

  // Fetch pre-computed sample hashes on mount
  useEffect(() => {
    fetchSampleHashes().then((h) => { sampleHashesRef.current = h; });
  }, []);

  const drawImage = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    const container = containerRef.current ?? canvas.parentElement!;
    const maxW = Math.min(container.clientWidth - 48, 860);
    const maxH = Math.min(container.clientHeight - 80, 640);
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
    const w = Math.floor(img.naturalWidth * scale);
    const h = Math.floor(img.naturalHeight * scale);

    canvas.width = w;
    canvas.height = h;
    for (const ref of [overlayARef, overlayBRef]) {
      if (ref.current) { ref.current.width = w; ref.current.height = h; }
    }
    ctx.drawImage(img, 0, 0, w, h);
    dimsRef.current = { scale, origW: img.naturalWidth, origH: img.naturalHeight };
  }, []);

  // Redraw canvas when container resizes (e.g. panel open/close)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      const img = imageRef.current;
      if (!img) return;
      drawImage(img);
      // Clear stale overlays and focus box
      for (const ref of [overlayARef, overlayBRef]) {
        if (ref.current) {
          ref.current.style.opacity = "0";
          const ctx = ref.current.getContext("2d");
          if (ctx) ctx.clearRect(0, 0, ref.current.width, ref.current.height);
        }
      }
      lastCategoryRef.current = -1;
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [drawImage]);

  // URL-driven search: handle ?s=<pointId> for find-similar
  useEffect(() => {
    const sParam = searchParams.get("s");
    if (!sParam) return;
    const pointId = parseInt(sParam, 10);
    if (isNaN(pointId)) return;

    similarIdRef.current = pointId;
    textSearchRef.current = null;
    offsetRef.current = 0;
    setHasMore(true);
    setSelectedCategory("similar");
    setSearching(true);
    setProducts([]);
    setFailedImages(new Set());

    findSimilar(pointId).then((sr) => {
      setSearchQuery(sr.query);
      setProducts(sr.products);
      setSearching(false);
      setHasMore(sr.products.length >= 100);
      offsetRef.current = sr.products.length;
    }).catch(() => setSearching(false));
  }, [searchParams]);

  const canvasToModelCoords = useCallback(
    (canvasX: number, canvasY: number) => {
      const { scale, origW, origH } = dimsRef.current;
      const origX = canvasX / scale;
      const origY = canvasY / scale;
      const maxDim = Math.max(origW, origH);
      const modelScale = 1024 / maxDim;
      return { x: origX * modelScale, y: origY * modelScale };
    },
    []
  );

  const processImage = useCallback(
    async (file: File, url: string) => {
      setImageSrc(url);
      setProducts([]); setFailedImages(new Set());
      setSearchQuery("");
      setSelectedCategory("");
      setFocusBox(null);
      setFocusDimmed(false);
      focusDimmedRef.current = false;
      fileRef.current = file;
      lastCategoryRef.current = -1;

      const img = new Image();
      img.onload = async () => {
        imageRef.current = img;
        drawImage(img);
        try {
          setStage("encoding");
          setStatusText("Analyzing outfit...");
          setProgress(0);

          let p = 0;
          const interval = setInterval(() => {
            p += (90 - p) * 0.08;
            setProgress(Math.round(p));
          }, 200);

          const result = await encodeImage(file);
          clearInterval(interval);
          setProgress(100);
          hashRef.current = result.hash;
          setStage("ready");
          setStatusText("Hover to detect — click to search");

          const canvas = canvasRef.current;
          if (canvas) {
            fetchClothingMask(result.hash, canvas.width, canvas.height)
              .then((r) => { if (r.mask) startDripRef.current?.(r.mask); })
              .catch(() => {});
          }
        } catch (err: any) {
          setStage("idle");
          setProgress(0);
          setStatusText(`Error: ${err.message}. Is the server running?`);
        }
      };
      img.src = url;
    },
    [drawImage]
  );

  const handleFile = useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file);
      processImage(file, url);
    },
    [processImage]
  );

  const handleSampleClick = useCallback(
    async (src: string) => {
      try {
        const cachedHash = sampleHashesRef.current[src];
        if (cachedHash) {
          // Pre-computed — skip encoding, go straight to ready
          setImageSrc(src);
          setProducts([]); setFailedImages(new Set());
          setSearchQuery("");
          setSelectedCategory("");
          setFocusBox(null);
          setFocusDimmed(false);
          focusDimmedRef.current = false;
          lastCategoryRef.current = -1;
          hashRef.current = cachedHash;

          const img = new window.Image();
          img.onload = () => {
            imageRef.current = img;
            drawImage(img);
            setStage("ready");
            setStatusText("Hover to detect — click to search");

            const canvas = canvasRef.current;
            if (canvas) {
              fetchClothingMask(cachedHash, canvas.width, canvas.height)
                .then((r) => { if (r.mask) startDripRef.current?.(r.mask); })
                .catch(() => {});
            }
          };
          img.src = src;
        } else {
          const res = await fetch(src);
          const blob = await res.blob();
          const file = new File([blob], src.split("/").pop() || "sample.jpg", { type: blob.type });
          processImage(file, src);
        }
      } catch (err: any) {
        setStatusText(`Error loading sample: ${err.message}`);
      }
    },
    [processImage, drawImage]
  );

  // --- Mask rendering ---
  const getOverlayRefs = useCallback(() => {
    const active = activeOverlayRef.current === "A" ? overlayARef : overlayBRef;
    const inactive = activeOverlayRef.current === "A" ? overlayBRef : overlayARef;
    return { active: active.current!, inactive: inactive.current! };
  }, []);

  const renderMask = useCallback(
    (maskB64: string, categoryId: number) => {
      if (clearTimerRef.current) { clearTimeout(clearTimerRef.current); clearTimerRef.current = null; }

      const { active: oldCanvas, inactive: newCanvas } = getOverlayRefs();

      const maskImg = new window.Image();
      maskImg.onload = () => {
        const tmp = document.createElement("canvas");
        tmp.width = newCanvas.width;
        tmp.height = newCanvas.height;
        const tmpCtx = tmp.getContext("2d")!;
        tmpCtx.drawImage(maskImg, 0, 0, newCanvas.width, newCanvas.height);
        const maskData = tmpCtx.getImageData(0, 0, newCanvas.width, newCanvas.height);
        maskImageDataRef.current = maskData;

        const ctx = newCanvas.getContext("2d")!;
        ctx.clearRect(0, 0, newCanvas.width, newCanvas.height);

        const focused = focusDimmedRef.current;
        if (focused) {
          const alphaMask = ctx.createImageData(newCanvas.width, newCanvas.height);
          for (let i = 0; i < maskData.data.length; i += 4) {
            const v = maskData.data[i];
            if (v > 20) {
              alphaMask.data[i] = 255;
              alphaMask.data[i + 1] = 255;
              alphaMask.data[i + 2] = 255;
              alphaMask.data[i + 3] = v;
            }
          }
          ctx.putImageData(alphaMask, 0, 0);
          ctx.globalCompositeOperation = "source-in";
          ctx.drawImage(canvasRef.current!, 0, 0);
          ctx.globalCompositeOperation = "source-over";
        } else {
          // Per-pixel adaptive brightening: bright areas get more, shadows stay natural
          const srcCtx = canvasRef.current!.getContext("2d")!;
          const srcData = srcCtx.getImageData(0, 0, newCanvas.width, newCanvas.height);

          const out = ctx.createImageData(newCanvas.width, newCanvas.height);
          for (let i = 0; i < maskData.data.length; i += 4) {
            const a = maskData.data[i];
            if (a > 20) {
              const n = Math.min(a / 255, 1);
              const soft = n * n * n;
              const lum = (srcData.data[i] * 0.299 + srcData.data[i + 1] * 0.587 + srcData.data[i + 2] * 0.114) / 255;
              // Shadows (~0): ~5%, midtones: ~15%, highlights (~1): ~35%
              const strength = 0.05 + lum * 0.15;
              out.data[i] = 255;
              out.data[i + 1] = 255;
              out.data[i + 2] = 255;
              out.data[i + 3] = Math.floor(soft * strength * 255);
            }
          }
          ctx.putImageData(out, 0, 0);
        }

        // Crossfade
        newCanvas.style.opacity = "0";
        requestAnimationFrame(() => {
          newCanvas.style.opacity = "1";
          oldCanvas.style.opacity = "0";
        });
        activeOverlayRef.current = activeOverlayRef.current === "A" ? "B" : "A";
      };
      maskImg.src = `data:image/png;base64,${maskB64}`;
    },
    [getOverlayRefs]
  );

  const clearOverlay = useCallback(() => {
    for (const ref of [overlayARef, overlayBRef]) {
      if (ref.current) ref.current.style.opacity = "0";
    }
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    clearTimerRef.current = setTimeout(() => {
      for (const ref of [overlayARef, overlayBRef]) {
        if (!ref.current) continue;
        const ctx = ref.current.getContext("2d")!;
        ctx.clearRect(0, 0, ref.current.width, ref.current.height);
      }
    }, 320);
  }, []);

  const startDrip = useCallback((maskB64: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    fullMaskB64Ref.current = maskB64;
    const w = canvas.width, h = canvas.height;

    const maskImg = new window.Image();
    maskImg.onload = () => {
      // Create feathered mask: blur edges + gamma curve
      const tmp = document.createElement("canvas");
      tmp.width = w; tmp.height = h;
      const ctx = tmp.getContext("2d")!;
      ctx.filter = "blur(8px)";
      ctx.drawImage(maskImg, 0, 0, w, h);
      ctx.filter = "none";

      const d = ctx.getImageData(0, 0, w, h);
      for (let i = 0; i < d.data.length; i += 4) {
        const v = d.data[i] / 255;
        d.data[i] = 255;
        d.data[i + 1] = 255;
        d.data[i + 2] = 255;
        d.data[i + 3] = Math.round(v * v * 255); // gamma exponent=2
      }
      ctx.putImageData(d, 0, 0);

      tmp.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        if (dripMaskUrlRef.current) URL.revokeObjectURL(dripMaskUrlRef.current);
        dripMaskUrlRef.current = url;
        setDripState({ maskUrl: url, width: w, height: h });
        setDripVisible(true);
      });
    };
    maskImg.src = `data:image/png;base64,${maskB64}`;
  }, []);
  startDripRef.current = startDrip;

  const stopDrip = useCallback(() => {
    setDripVisible(false);
  }, []);

  const buildFocusMask = useCallback((garmentB64: string, w: number, h: number) => {
    const fullB64 = fullMaskB64Ref.current;
    if (!fullB64) return;
    const loadImg = (src: string): Promise<HTMLImageElement> => new Promise((res) => {
      const img = new window.Image();
      img.onload = () => res(img);
      img.src = `data:image/png;base64,${src}`;
    });
    Promise.all([loadImg(fullB64), loadImg(garmentB64)]).then(([fullImg, garmentImg]) => {
      // Draw full clothing mask, subtract selected garment
      const tmp = document.createElement("canvas");
      tmp.width = w; tmp.height = h;
      const ctx = tmp.getContext("2d")!;
      ctx.drawImage(fullImg, 0, 0, w, h);
      const fd = ctx.getImageData(0, 0, w, h);
      const gTmp = document.createElement("canvas");
      gTmp.width = w; gTmp.height = h;
      const gCtx = gTmp.getContext("2d")!;
      gCtx.drawImage(garmentImg, 0, 0, w, h);
      const gd = gCtx.getImageData(0, 0, w, h);
      for (let i = 0; i < fd.data.length; i += 4) {
        if (gd.data[i] > 20) { fd.data[i] = 0; fd.data[i + 1] = 0; fd.data[i + 2] = 0; }
      }
      ctx.putImageData(fd, 0, 0);

      // Blur for feathered edges, then gamma
      const tmp2 = document.createElement("canvas");
      tmp2.width = w; tmp2.height = h;
      const ctx2 = tmp2.getContext("2d")!;
      ctx2.filter = "blur(8px)";
      ctx2.drawImage(tmp, 0, 0);
      ctx2.filter = "none";
      const d = ctx2.getImageData(0, 0, w, h);
      for (let i = 0; i < d.data.length; i += 4) {
        const v = d.data[i] / 255;
        d.data[i] = 255; d.data[i + 1] = 255; d.data[i + 2] = 255;
        d.data[i + 3] = Math.round(v * v * 255);
      }
      ctx2.putImageData(d, 0, 0);

      tmp2.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        if (focusMaskUrlRef.current) URL.revokeObjectURL(focusMaskUrlRef.current);
        focusMaskUrlRef.current = url;
        setFocusMaskUrl(url);
      });
    });
  }, []);

  const dismissFocus = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      setFocusBox({ x: 0, y: 0, w: canvas.width, h: canvas.height });
    }
    setFocusDimmed(false);
    focusDimmedRef.current = false;
    setFocusMaskUrl(null);
    if (focusMaskUrlRef.current) { URL.revokeObjectURL(focusMaskUrlRef.current); focusMaskUrlRef.current = null; }
    setTimeout(() => setFocusBox(null), 400);
    // Restart drip if it was cleared (e.g. by reset)
    if (!dripMaskUrlRef.current && hashRef.current && canvas) {
      fetchClothingMask(hashRef.current, canvas.width, canvas.height)
        .then((r) => { if (r.mask) startDrip(r.mask); })
        .catch(() => {});
    }
  }, [startDrip]);

  const handleMouseLeave = useCallback(() => {
    clearOverlay();
    lastCategoryRef.current = -1;
  }, [clearOverlay]);

  const handleHover = useCallback(
    async (e: React.MouseEvent<HTMLElement>) => {
      if (stage !== "ready" || !hashRef.current) return;
      const now = Date.now();
      if (now - hoverThrottleRef.current < 100) return;
      hoverThrottleRef.current = now;

      const rect = canvasRef.current!.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      if (focusDimmedRef.current && focusBox) {
        if (cx >= focusBox.x && cx <= focusBox.x + focusBox.w &&
            cy >= focusBox.y && cy <= focusBox.y + focusBox.h) {
          if (lastCategoryRef.current !== -1) {
            lastCategoryRef.current = -1;
            clearOverlay();
          }
          return;
        }
      }

      const { x, y } = canvasToModelCoords(cx, cy);
      const canvas = canvasRef.current!;

      try {
        const result = await decodeMask(hashRef.current, x, y, canvas.width, canvas.height);
        if (result.mask) {
          if (result.categoryId === lastCategoryRef.current) return;
          lastCategoryRef.current = result.categoryId;
          lastHoverResultRef.current = { mask: result.mask, category: result.category, categoryId: result.categoryId };
          renderMask(result.mask, result.categoryId);
        } else {
          lastCategoryRef.current = -1;
          lastHoverResultRef.current = null;
          clearOverlay();
        }
      } catch {}
    },
    [stage, canvasToModelCoords, renderMask, clearOverlay, focusBox]
  );

  const animateFocusBox = useCallback((bbox: { x: number; y: number; w: number; h: number }) => {
    const canvas = canvasRef.current!;
    const inset = 14;
    const maxW = Math.min(bbox.w, canvas.width - inset * 2);
    const maxH = Math.min(bbox.h, canvas.height - inset * 2);
    const clamped = {
      x: Math.max(inset, Math.min(bbox.x, canvas.width - maxW - inset)),
      y: Math.max(inset, Math.min(bbox.y, canvas.height - maxH - inset)),
      w: maxW,
      h: maxH,
    };
    if (!focusDimmedRef.current) {
      setFocusBox({ x: 0, y: 0, w: canvas.width, h: canvas.height });
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setFocusBox(clamped);
          setFocusDimmed(true);
          focusDimmedRef.current = true;
        });
      });
    } else {
      setFocusBox(clamped);
    }
  }, []);

  const decodeMaskToBbox = useCallback((maskB64: string, canvasW: number, canvasH: number): Promise<{ x: number; y: number; w: number; h: number } | null> => {
    return new Promise((resolve) => {
      const maskImg = new window.Image();
      maskImg.onload = () => {
        const tmp = document.createElement("canvas");
        tmp.width = canvasW;
        tmp.height = canvasH;
        const tmpCtx = tmp.getContext("2d")!;
        tmpCtx.drawImage(maskImg, 0, 0, canvasW, canvasH);
        const maskData = tmpCtx.getImageData(0, 0, canvasW, canvasH);
        resolve(computeBboxFromMask(maskData));
      };
      maskImg.src = `data:image/png;base64,${maskB64}`;
    });
  }, []);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLElement>) => {
      if (stage !== "ready" || !hashRef.current) return;
      const rect = canvasRef.current!.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      if (focusDimmedRef.current && focusBox) {
        if (canvasX >= focusBox.x && canvasX <= focusBox.x + focusBox.w &&
            canvasY >= focusBox.y && canvasY <= focusBox.y + focusBox.h) {
          dismissFocus();
          clearOverlay();
          setProducts([]); setFailedImages(new Set());
          setSearchQuery("");
          setSelectedCategory("");
          lastCategoryRef.current = -1;
          return;
        }
      }
      const { x, y } = canvasToModelCoords(canvasX, canvasY);
      const canvas = canvasRef.current!;
      const hash = hashRef.current;

      const gen = ++clickGenRef.current;

      clearOverlay();
      setSearching(true);
      setProducts([]); setFailedImages(new Set());
      setSearchQuery("");

      decodeMask(hash, x, y, canvas.width, canvas.height).then((result) => {
        if (clickGenRef.current !== gen) return;

        if (result.mask) {
          decodeMaskToBbox(result.mask, canvas.width, canvas.height).then((bbox) => {
            if (clickGenRef.current !== gen) return;
            if (bbox) animateFocusBox(bbox);
          });
          buildFocusMask(result.mask, canvas.width, canvas.height);

          setSelectedCategory(result.category);
          textSearchRef.current = { hash, categoryId: result.categoryId };
          similarIdRef.current = null;
          offsetRef.current = 0;
          setHasMore(true);

          searchProducts(hash, result.categoryId).then((sr) => {
            if (clickGenRef.current !== gen) return;
            setSearchQuery(sr.query);
            setProducts(sr.products);
            setSearching(false);
            setHasMore(sr.products.length >= 100);
            offsetRef.current = sr.products.length;
            // Push URL state
            const params = new URLSearchParams();
            params.set("q", sr.query);
            params.set("cat", String(result.categoryId));
            router.push(`/?${params.toString()}`, { scroll: false });
          }).catch(() => {
            if (clickGenRef.current === gen) setSearching(false);
          });
        } else {
          const size = 120;
          const bx = Math.max(0, Math.min(canvas.width - size, canvasX - size / 2));
          const by = Math.max(0, Math.min(canvas.height - size, canvasY - size / 2));
          animateFocusBox({ x: bx, y: by, w: size, h: size });
          setSelectedCategory("");
          setSearching(false);
        }
      }).catch(() => {
        if (clickGenRef.current === gen) setSearching(false);
      });
    },
    [stage, canvasToModelCoords, clearOverlay, animateFocusBox, decodeMaskToBbox, focusBox, dismissFocus, buildFocusMask]
  );

  // Handle clicking a product card → find similar
  const handleProductClick = useCallback((product: Product) => {
    similarIdRef.current = product.id;
    textSearchRef.current = null;
    offsetRef.current = 0;
    setHasMore(true);
    setSelectedCategory("similar");
    setSearching(true);
    setProducts([]);
    setFailedImages(new Set());

    router.push(`/?s=${product.id}`, { scroll: false });

    findSimilar(product.id).then((sr) => {
      setSearchQuery(sr.query);
      setProducts(sr.products);
      setSearching(false);
      setHasMore(sr.products.length >= 100);
      offsetRef.current = sr.products.length;
    }).catch(() => setSearching(false));
  }, [router]);

  // Infinite scroll: load more when sentinel enters viewport
  const loadMore = useCallback(() => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);

    const fetchNext = similarIdRef.current != null
      ? findSimilar(similarIdRef.current, 100, offsetRef.current)
      : textSearchRef.current
        ? searchProducts(textSearchRef.current.hash, textSearchRef.current.categoryId, 100, offsetRef.current)
        : null;

    if (!fetchNext) { setLoadingMore(false); return; }

    fetchNext.then((sr) => {
      setProducts((prev) => [...prev, ...sr.products]);
      offsetRef.current += sr.products.length;
      setHasMore(sr.products.length >= 100);
      setLoadingMore(false);
    }).catch(() => setLoadingMore(false));
  }, [loadingMore, hasMore]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) loadMore(); },
      { rootMargin: "400px" }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMore]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  const reset = () => {
    stopDrip();
    setDripState(null);
    if (dripMaskUrlRef.current) { URL.revokeObjectURL(dripMaskUrlRef.current); dripMaskUrlRef.current = null; }
    setFocusMaskUrl(null);
    if (focusMaskUrlRef.current) { URL.revokeObjectURL(focusMaskUrlRef.current); focusMaskUrlRef.current = null; }
    fullMaskB64Ref.current = null;
    setImageSrc(null);
    setStage("idle");
    setStatusText("");
    setProducts([]); setFailedImages(new Set());
    setSearchQuery("");
    setSelectedCategory("");
    setFocusBox(null);
    setFocusDimmed(false);
    focusDimmedRef.current = false;
    hashRef.current = "";
    similarIdRef.current = null;
    textSearchRef.current = null;
    setLandingKey((k) => k + 1);
    router.push("/", { scroll: false });
  };

  const closePanel = useCallback(() => {
    setProducts([]); setFailedImages(new Set());
    setSearchQuery("");
    setSelectedCategory("");
    dismissFocus();
    clearOverlay();
    lastCategoryRef.current = -1;
  }, [dismissFocus, clearOverlay]);

  return (
    <div className="min-h-screen bg-[#F8F7F4] text-[#1a1a1a] relative">
      {/* Header */}
      <header className="relative z-10 px-6 py-5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <button onClick={imageSrc ? reset : undefined} className="group">
            <h1 className="text-[17px] font-medium tracking-tight text-[#1a1a1a]">
              Closet Drift
            </h1>
          </button>
          {imageSrc && (
            <button onClick={reset} className="text-xs text-[#999] hover:text-[#1a1a1a]  rounded-full px-4 py-1.5 transition-all duration-200">
              New image
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="relative z-10 flex flex-col lg:flex-row" style={{ height: "calc(100vh - 65px)" }}>
        {/* Left: Image */}
        <div ref={containerRef} className={`flex-1 min-w-0 flex flex-col items-center overflow-auto ${imageSrc ? "justify-start p-6" : "justify-center p-6"}`}>
          {!imageSrc ? (
            <div key={landingKey} className="flex flex-col items-center w-full max-w-3xl animate-fade-in" ref={(el) => { if (el) console.log("[landing] wrapper mounted, offsetWidth:", el.offsetWidth, "offsetHeight:", el.offsetHeight); }}>
              {/* Hero text */}
              <div className="text-center pt-6 pb-6 shrink-0">
                <h2 className="text-5xl font-light tracking-tight text-[#1a1a1a] leading-[1.1]">
                  Find what you love.
                </h2>
              </div>

              {/* Sample images — masonry */}
              <div className="w-full py-6" ref={(el) => { if (el) console.log("[gallery] container mounted, offsetWidth:", el.offsetWidth, "offsetHeight:", el.offsetHeight, "children:", el.children.length, "innerHTML length:", el.innerHTML.length); }}>
                <MasonryPhotoAlbum
                  photos={SAMPLE_PHOTOS}
                  columns={3}
                  spacing={8}
                  onClick={({ photo }) => handleSampleClick(photo.src)}
                />
              </div>

              {/* Upload area */}
              <label
                className={`shrink-0 group relative flex items-center justify-center w-full max-w-sm h-12 rounded-full cursor-pointer transition-all duration-300 ${
                  dragging
                    ? "border-[#d4d0c8] bg-[#f0efe9] border-dashed border"
                    : "border-[#d4d0c8] hover:border-[#aaa]"
                }`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { setDragging(false); handleDrop(e); }}
              >
                <span className="text-[13px] text-[#999] group-hover:text-[#666] transition-colors">
                  or upload your own
                </span>
                <input type="file" className="hidden" accept="image/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </label>
            </div>
          ) : (
            <div className="flex flex-col items-center w-full animate-fade-in-scale">
                {/* Canvas stack */}
              <div className="relative inline-block rounded-2xl overflow-hidden shadow-lg shadow-black/8 ring-1 ring-black/5">
                <canvas ref={canvasRef} className="block" />

                {/* Progress overlay removed — analyzing message now below image */}
                {/* Pinterest-style glow sweep */}
                {dripState && (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      zIndex: focusDimmed ? 11 : 1,
                      opacity: dripVisible ? 1 : 0,
                      transition: "opacity 1.5s ease, z-index 0s",
                    }}
                  >
                    <style>{`
                      @keyframes ${dripId}-sweep {
                        0%   { -webkit-mask-position: 0 ${Math.round(-dripState.height * 0.675)}px; mask-position: 0 ${Math.round(-dripState.height * 0.675)}px; opacity: 0; }
                        3%   { opacity: 0.5; }
                        33%  { -webkit-mask-position: 0 ${Math.round(dripState.height * 0.675)}px; mask-position: 0 ${Math.round(dripState.height * 0.675)}px; opacity: 0.5; }
                        34%  { -webkit-mask-position: 0 ${Math.round(dripState.height * 0.675)}px; mask-position: 0 ${Math.round(dripState.height * 0.675)}px; opacity: 0; }
                        100% { -webkit-mask-position: 0 ${Math.round(-dripState.height * 0.675)}px; mask-position: 0 ${Math.round(-dripState.height * 0.675)}px; opacity: 0; }
                      }
                    `}</style>
                    <div
                      className="absolute inset-0"
                      style={{
                        maskImage: `url(${focusDimmed && focusMaskUrl ? focusMaskUrl : dripState.maskUrl})`,
                        WebkitMaskImage: `url(${focusDimmed && focusMaskUrl ? focusMaskUrl : dripState.maskUrl})`,
                        maskSize: "100% 100%",
                        WebkitMaskSize: "100% 100%",
                        transition: "mask-image 0.3s ease, -webkit-mask-image 0.3s ease",
                      }}
                    >
                      <div
                        className="absolute inset-0"
                        style={{
                          animation: `${dripId}-sweep 9s linear 5s infinite backwards`,
                          maskImage: `radial-gradient(${Math.round(dripState.width * 2.5)}px ${Math.round(dripState.height * 0.35)}px, white 0%, white 15%, rgba(255,255,255,0.9) 30%, rgba(255,255,255,0.7) 45%, rgba(255,255,255,0.5) 55%, rgba(255,255,255,0.3) 65%, rgba(255,255,255,0.15) 72%, rgba(255,255,255,0.06) 80%, rgba(255,255,255,0.01) 90%, transparent 100%)`,
                          WebkitMaskImage: `radial-gradient(${Math.round(dripState.width * 2.5)}px ${Math.round(dripState.height * 0.35)}px, white 0%, white 15%, rgba(255,255,255,0.9) 30%, rgba(255,255,255,0.7) 45%, rgba(255,255,255,0.5) 55%, rgba(255,255,255,0.3) 65%, rgba(255,255,255,0.15) 72%, rgba(255,255,255,0.06) 80%, rgba(255,255,255,0.01) 90%, transparent 100%)`,
                          maskRepeat: "no-repeat",
                          WebkitMaskRepeat: "no-repeat",
                          maskSize: "100% 100%",
                          WebkitMaskSize: "100% 100%",
                        } as React.CSSProperties}
                      >
                        {/* Rainbow glow when idle, white fog when garment focused */}
                        <div
                          className="absolute inset-0"
                          style={{
                            background: focusDimmed
                              ? "white"
                              : "linear-gradient(135deg, #a0c4f0 0%, #c8a0e8 18%, #f0a0a0 32%, #f0c878 48%, #e0e0a0 58%, #80d8b8 72%, #a0c8e8 88%, #c8a0d8 100%)",
                            transition: "background 0.3s ease",
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}
                <canvas ref={overlayARef} className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-300 ease-out" style={{ zIndex: 11 }} />
                <canvas ref={overlayBRef} className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-300 ease-out" style={{ zIndex: 11 }} />
                {/* Focus box overlay */}
                {focusBox && (
                  <div
                    className="absolute pointer-events-none focus-box-transition"
                    style={{
                      left: focusBox.x,
                      top: focusBox.y,
                      width: focusBox.w,
                      height: focusBox.h,
                      borderRadius: 12,
                      boxShadow: focusDimmed
                        ? "0 0 0 9999px rgba(0, 0, 0, 0.35)"
                        : "0 0 0 9999px rgba(0, 0, 0, 0)",
                      zIndex: 10,
                    }}
                  >
                    <svg
                      className="absolute focus-box-transition"
                      style={{ top: -10, right: -10, opacity: focusDimmed ? 1 : 0, zIndex: 12 }}
                      width="24" height="24" viewBox="0 0 24 24" fill="none"
                    >
                      <path d="M2 2A20 20 0 0 1 22 22" stroke="rgba(255,255,255,0.85)" strokeWidth="4.5" strokeLinecap="round" />
                    </svg>
                    <svg
                      className="absolute focus-box-transition"
                      style={{ bottom: -10, left: -10, opacity: focusDimmed ? 1 : 0, zIndex: 12 }}
                      width="24" height="24" viewBox="0 0 24 24" fill="none"
                    >
                      <path d="M22 22A20 20 0 0 1 2 2" stroke="rgba(255,255,255,0.85)" strokeWidth="4.5" strokeLinecap="round" />
                    </svg>
                  </div>
                )}

                <div
                  className="absolute inset-0"
                  style={{ cursor: stage === "ready" ? "pointer" : "default", zIndex: 20 }}
                  onMouseMove={handleHover}
                  onMouseLeave={handleMouseLeave}
                  onClick={handleClick}
                />
              </div>

              {/* Error message below the image */}
              {stage === "idle" && statusText && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-2.5 mt-4">{statusText}</div>
              )}

              {/* Analyzing message below the image */}
              {stage === "encoding" && (
                <div className="flex flex-col items-center gap-2 mt-4 w-full max-w-[320px] mx-auto">
                  <div className="flex items-center justify-between w-full">
                    <span className="text-sm text-[#666]">{statusText}</span>
                    <span className="text-xs text-[#aaa] tabular-nums">{progress}%</span>
                  </div>
                  <div className="w-full h-1 bg-[#e8e5df] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#1a1a1a] rounded-full transition-all duration-200 ease-out"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Product panel */}
        <div
          style={{ containerType: "inline-size" }}
          className={`bg-[#F8F7F4] overflow-y-auto transition-all duration-500 ease-in-out shrink-0 hidden lg:block ${
            selectedCategory ? "lg:w-[50vw] border-[#e8e5df]" : "lg:w-0 lg:border-0 lg:overflow-hidden"
          }`}
        >
          {selectedCategory && (
            <div className="p-6 animate-fade-in">
              {/* Skeleton loading */}
              {searching && (
                <div className="skeleton-masonry">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div
                      key={i}
                      className="rounded-lg bg-[#EDEAE5] animate-pulse"
                      style={{ height: [180, 260, 200, 240, 190, 280, 220, 250, 170, 230, 260, 210][i] }}
                    />
                  ))}
                </div>
              )}

              {/* No results */}
              {!searching && products.length === 0 && searchQuery && (
                <div className="text-center py-16">
                  <p className="text-sm text-[#999]">No products found</p>
                </div>
              )}

              {/* Product masonry */}
              {products.length > 0 && (
                <MasonryPhotoAlbum
                  photos={products
                    .filter((p) => !failedImages.has(p.image) && p.image)
                    .map((p) => ({
                      src: p.image.startsWith("/") ? `${API_BASE}${p.image}` : p.image,
                      width: p.width || 300,
                      height: p.height || 400,
                      key: `${p.id}`,
                      title: p.name,
                    }))}
                  columns={(containerWidth) => {
                    if (containerWidth < 250) return 2;
                    if (containerWidth < 400) return 3;
                    if (containerWidth < 600) return 4;
                    if (containerWidth < 800) return 5;
                    return 6;
                  }}
                  spacing={6}
                  onClick={({ index }) => {
                    const visible = products.filter((p) => !failedImages.has(p.image) && p.image);
                    const product = visible[index];
                    if (product) handleProductClick(product);
                  }}
                  render={{
                    image: (props) => (
                      /* eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text */
                      <img
                        {...props}
                        loading="lazy"
                        decoding="async"
                        className={`${props.className || ""} product-img rounded-lg`}
                        onLoad={(e) => e.currentTarget.classList.add("loaded")}
                        onError={(e) => {
                          const src = e.currentTarget.src;
                          setFailedImages((prev) => new Set(prev).add(
                            src.startsWith(API_BASE) ? src.slice(API_BASE.length) : src
                          ));
                        }}
                      />
                    ),
                  }}
                />
              )}

              {/* Infinite scroll sentinel */}
              {hasMore && products.length > 0 && (
                <div ref={sentinelRef} className="flex justify-center py-8">
                  {loadingMore && (
                    <div className="w-6 h-6 border-2 border-[#d4d0c8] border-t-[#1a1a1a] rounded-full animate-spin" />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
