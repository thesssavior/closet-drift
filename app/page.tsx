"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { MasonryPhotoAlbum } from "react-photo-album";
import "react-photo-album/masonry.css";
import {
  encodeImage,
  decodeMask,
  searchProducts,
  fetchClothingMask,
  API_BASE,
  type Product,
} from "./lib/sam2";

type Stage = "idle" | "encoding" | "ready";

const SAMPLE_PHOTOS = [
  { src: "/samples/1.jpeg", width: 736, height: 1104 },
  { src: "/samples/2.jpg", width: 1707, height: 2560 },
  { src: "/samples/3.jpg", width: 540, height: 360 },
  { src: "/samples/4.jpg", width: 1110, height: 1665 },
  { src: "/samples/5.jpg", width: 1024, height: 683 },
  { src: "/samples/6.jpg", width: 1920, height: 1280 },
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

export default function Home() {
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
  const dripRef = useRef<HTMLCanvasElement>(null);
  const dripMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dripRafRef = useRef<number>(0);
  const startDripRef = useRef<((maskB64: string) => void) | null>(null);

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
    for (const ref of [overlayARef, overlayBRef, dripRef]) {
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
        const res = await fetch(src);
        const blob = await res.blob();
        const file = new File([blob], src.split("/").pop() || "sample.jpg", { type: blob.type });
        processImage(file, src);
      } catch (err: any) {
        setStatusText(`Error loading sample: ${err.message}`);
      }
    },
    [processImage]
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
              const soft = n * n * n * n;
              const lum = (srcData.data[i] * 0.299 + srcData.data[i + 1] * 0.587 + srcData.data[i + 2] * 0.114) / 255;
              // Shadows (~0): ~2%, midtones: ~5%, highlights (~1): ~10%
              const strength = 0.02 + lum * lum * 0.3;
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
    const drip = dripRef.current;
    if (!drip) return;

    const maskImg = new window.Image();
    maskImg.onload = () => {
      const w = drip.width;
      const h = drip.height;

      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = w;
      maskCanvas.height = h;
      const maskCtx = maskCanvas.getContext("2d")!;
      maskCtx.drawImage(maskImg, 0, 0, w, h);
      const raw = maskCtx.getImageData(0, 0, w, h);
      for (let i = 0; i < raw.data.length; i += 4) {
        const v = raw.data[i];
        raw.data[i] = 255;
        raw.data[i + 1] = 255;
        raw.data[i + 2] = 255;
        raw.data[i + 3] = v > 20 ? v : 0;
      }
      maskCtx.putImageData(raw, 0, 0);
      dripMaskCanvasRef.current = maskCanvas;

      const gradCanvas = document.createElement("canvas");
      gradCanvas.width = w;
      gradCanvas.height = h;
      const gradCtx = gradCanvas.getContext("2d")!;
      const baseGrad = gradCtx.createLinearGradient(0, 0, w, h);
      baseGrad.addColorStop(0, "rgba(120, 60, 255, 0.25)");
      baseGrad.addColorStop(0.2, "rgba(80, 120, 255, 0.22)");
      baseGrad.addColorStop(0.4, "rgba(60, 200, 220, 0.18)");
      baseGrad.addColorStop(0.6, "rgba(80, 220, 120, 0.16)");
      baseGrad.addColorStop(0.8, "rgba(255, 200, 60, 0.18)");
      baseGrad.addColorStop(1, "rgba(255, 80, 80, 0.25)");
      gradCtx.fillStyle = baseGrad;
      gradCtx.fillRect(0, 0, w, h);
      gradCtx.globalCompositeOperation = "destination-in";
      gradCtx.drawImage(maskCanvas, 0, 0);
      gradCtx.globalCompositeOperation = "source-over";

      drip.style.opacity = "1";
      const ctx = drip.getContext("2d")!;
      const radius = Math.max(w, h) * 0.9;
      let start: number | null = null;
      const sweepDuration = 6000;
      const pauseDuration = 10000;
      const cycleDuration = sweepDuration + pauseDuration;

      const blobCanvas = document.createElement("canvas");
      blobCanvas.width = w;
      blobCanvas.height = h;

      const animate = (ts: number) => {
        if (!start) start = ts;
        const elapsed = ((ts - start) % cycleDuration);

        if (elapsed >= sweepDuration) {
          ctx.clearRect(0, 0, w, h);
          dripRafRef.current = requestAnimationFrame(animate);
          return;
        }

        const raw = elapsed / sweepDuration;
        const t = raw < 0.5
          ? 2 * raw * raw
          : 1 - 2 * (1 - raw) * (1 - raw);
        const cx = -radius * 0.2 + (w + radius * 0.4) * t;
        const cy = -radius * 0.2 + (h + radius * 0.4) * t;
        const edgeFade = Math.min(raw / 0.2, 1) * Math.min((1 - raw) / 0.2, 1);
        const smoothFade = edgeFade * edgeFade * (3 - 2 * edgeFade);

        const bCtx = blobCanvas.getContext("2d")!;
        bCtx.clearRect(0, 0, w, h);
        const fog = bCtx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        fog.addColorStop(0, "rgba(255,255,255,1)");
        fog.addColorStop(0.3, "rgba(255,255,255,0.8)");
        fog.addColorStop(0.6, "rgba(255,255,255,0.3)");
        fog.addColorStop(0.85, "rgba(255,255,255,0.05)");
        fog.addColorStop(1, "rgba(255,255,255,0)");
        bCtx.fillStyle = fog;
        bCtx.fillRect(0, 0, w, h);

        ctx.clearRect(0, 0, w, h);
        ctx.globalAlpha = smoothFade;
        ctx.drawImage(gradCanvas, 0, 0);

        ctx.globalCompositeOperation = "destination-in";
        ctx.drawImage(blobCanvas, 0, 0);
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;

        dripRafRef.current = requestAnimationFrame(animate);
      };
      dripRafRef.current = requestAnimationFrame(animate);
    };
    maskImg.src = `data:image/png;base64,${maskB64}`;
  }, []);
  startDripRef.current = startDrip;

  const stopDrip = useCallback(() => {
    if (dripRafRef.current) {
      cancelAnimationFrame(dripRafRef.current);
      dripRafRef.current = 0;
    }
    const drip = dripRef.current;
    if (drip) {
      drip.style.opacity = "0";
      setTimeout(() => {
        const ctx = drip.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, drip.width, drip.height);
      }, 300);
    }
  }, []);

  const dismissFocus = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      setFocusBox({ x: 0, y: 0, w: canvas.width, h: canvas.height });
    }
    setFocusDimmed(false);
    focusDimmedRef.current = false;
    setTimeout(() => setFocusBox(null), 400);
    if (hashRef.current && canvas) {
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

      stopDrip();
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

          setSelectedCategory(result.category);

          searchProducts(hash, result.categoryId).then((sr) => {
            if (clickGenRef.current !== gen) return;
            setSearchQuery(sr.query);
            setProducts(sr.products);
            setSearching(false);
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
    [stage, canvasToModelCoords, clearOverlay, animateFocusBox, decodeMaskToBbox, focusBox, dismissFocus, stopDrip]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  const reset = () => {
    console.log("[reset] called, current imageSrc:", imageSrc);
    stopDrip();
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
    setLandingKey((k) => k + 1);
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
            <button onClick={reset} className="text-xs text-[#999] hover:text-[#1a1a1a] border border-[#e0ddd8] hover:border-[#ccc] rounded-full px-4 py-1.5 transition-all duration-200">
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
                  Find what you like.
                </h2>
              </div>

              {/* Sample images — masonry */}
              <div className="w-full py-8" ref={(el) => { if (el) console.log("[gallery] container mounted, offsetWidth:", el.offsetWidth, "offsetHeight:", el.offsetHeight, "children:", el.children.length, "innerHTML length:", el.innerHTML.length); }}>
                <MasonryPhotoAlbum
                  photos={SAMPLE_PHOTOS}
                  columns={3}
                  spacing={8}
                  onClick={({ photo }) => handleSampleClick(photo.src)}
                />
              </div>

              {/* Upload area */}
              <label
                className={`shrink-0 group relative flex items-center justify-center w-full max-w-sm h-12 border border-dashed rounded-full cursor-pointer transition-all duration-300 ${
                  dragging
                    ? "border-[#1a1a1a] bg-[#f0efe9]"
                    : "border-[#d4d0c8] hover:border-[#aaa] hover:bg-[#f0efe9]"
                }`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { setDragging(false); handleDrop(e); }}
              >
                <span className="text-[13px] text-[#999] group-hover:text-[#666] transition-colors">
                  or drop your own
                </span>
                <input type="file" className="hidden" accept="image/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </label>
            </div>
          ) : (
            <div className="flex flex-col items-center w-full animate-fade-in-scale">
              {/* Status */}
              {stage === "encoding" && (
                <div className="flex flex-col items-center gap-2 bg-white border border-[#e8e5df] rounded-xl px-5 py-3 shadow-sm min-w-[260px] mb-5">
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

              {stage === "idle" && statusText && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-2.5 mb-5">{statusText}</div>
              )}

              {/* Canvas stack */}
              <div className="relative inline-block rounded-2xl overflow-hidden shadow-lg shadow-black/8 ring-1 ring-black/5">
                <canvas ref={canvasRef} className="block" />
                <canvas ref={dripRef} className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-300 ease-out" style={{ zIndex: 1, opacity: 0 }} />
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
                <div className="masonry-tight">
                  {Array.from({ length: 9 }).map((_, i) => (
                    <div
                      key={i}
                      className="rounded-lg bg-[#EDEAE5] animate-pulse"
                      style={{ height: [180, 240, 200, 260, 190, 230, 210, 250, 220][i] }}
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
              <div className="masonry-tight stagger-children">
                {products.filter((p) => !failedImages.has(p.image)).map((product, i) => (
                  <div
                    key={`${product.id}-${i}`}
                    className="group rounded-lg overflow-hidden cursor-pointer"
                  >
                    {product.image && (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img
                        src={product.image.startsWith("/") ? `${API_BASE}${product.image}` : product.image}
                        alt={product.name}
                        className="product-img w-full h-auto block rounded-lg group-hover:brightness-[0.92] transition-[filter] duration-300"
                        loading="lazy"
                        onLoad={(e) => e.currentTarget.classList.add("loaded")}
                        onError={() => { setFailedImages((prev) => new Set(prev).add(product.image)); }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
