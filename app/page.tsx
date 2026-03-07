"use client";

import { useState, useRef, useCallback } from "react";
import {
  encodeImage,
  decodeMask,
  searchProducts,
  type Product,
} from "./lib/sam2";

type Stage = "idle" | "encoding" | "ready";

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
  const [products, setProducts] = useState<Product[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [focusBox, setFocusBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [focusDimmed, setFocusDimmed] = useState(false);

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

  const drawImage = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current!;
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

  const handleFile = useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file);
      setImageSrc(url);
      setProducts([]);
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
          // Build alpha mask from the R channel, then composite original image through it
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
          const out = ctx.createImageData(newCanvas.width, newCanvas.height);
          for (let i = 0; i < maskData.data.length; i += 4) {
            const a = maskData.data[i];
            if (a > 20) {
              const n = Math.min(a / 255, 1);
              const soft = n * n * n * n;
              out.data[i] = 255;
              out.data[i + 1] = 245;
              out.data[i + 2] = 235;
              out.data[i + 3] = Math.floor(soft * 0.03 * 255);
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

  const dismissFocus = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      setFocusBox({ x: 0, y: 0, w: canvas.width, h: canvas.height });
    }
    setFocusDimmed(false);
    focusDimmedRef.current = false;
    setTimeout(() => setFocusBox(null), 400);
  }, []);

  const handleMouseLeave = useCallback(() => {
    clearOverlay();
    lastCategoryRef.current = -1;
  }, [clearOverlay]);

  // --- Hover / Click ---
  const handleHover = useCallback(
    async (e: React.MouseEvent<HTMLElement>) => {
      if (stage !== "ready" || !hashRef.current) return;
      const now = Date.now();
      if (now - hoverThrottleRef.current < 100) return;
      hoverThrottleRef.current = now;

      const rect = canvasRef.current!.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      // If focused and cursor is inside the focus box, do nothing
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
    // Clamp box within canvas bounds with inset so corners don't touch edges
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

      // Click inside the focus box → dismiss it
      if (focusDimmedRef.current && focusBox) {
        if (canvasX >= focusBox.x && canvasX <= focusBox.x + focusBox.w &&
            canvasY >= focusBox.y && canvasY <= focusBox.y + focusBox.h) {
          dismissFocus();
          clearOverlay();
          setProducts([]);
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
      setProducts([]);
      setSearchQuery("");

      decodeMask(hash, x, y, canvas.width, canvas.height).then((result) => {
        if (clickGenRef.current !== gen) return; // stale click, discard

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
    [stage, canvasToModelCoords, clearOverlay, animateFocusBox, decodeMaskToBbox, focusBox, dismissFocus]
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
    setImageSrc(null);
    setStage("idle");
    setStatusText("");
    setProducts([]);
    setSearchQuery("");
    setSelectedCategory("");
    setFocusBox(null);
    setFocusDimmed(false);
    focusDimmedRef.current = false;
    hashRef.current = "";
  };

  const closePanel = useCallback(() => {
    setProducts([]);
    setSearchQuery("");
    setSelectedCategory("");
    dismissFocus();
    clearOverlay();
    lastCategoryRef.current = -1;
  }, [dismissFocus, clearOverlay]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 relative overflow-hidden">
      {/* Ambient blobs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-60 -left-60 w-[500px] h-[500px] bg-purple-600/[0.04] rounded-full blur-[120px]" />
        <div className="absolute -bottom-60 -right-60 w-[500px] h-[500px] bg-blue-600/[0.04] rounded-full blur-[120px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-500/[0.02] rounded-full blur-[150px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/[0.06] px-6 py-4 backdrop-blur-xl bg-zinc-950/70">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight text-white">Fit Detector</h1>
              <p className="text-[11px] text-zinc-500">AI-powered outfit recognition</p>
            </div>
          </div>
          {imageSrc && (
            <button onClick={reset} className="text-xs text-zinc-500 hover:text-zinc-200 border border-white/[0.08] hover:border-white/[0.15] hover:bg-white/[0.04] rounded-lg px-3 py-1.5 transition-all duration-200">
              New image
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="relative z-10 flex flex-col lg:flex-row h-[calc(100vh-57px)]">
        {/* Left: Image */}
        <div ref={containerRef} className="flex-1 flex flex-col items-center justify-center p-6 overflow-auto">
          {!imageSrc ? (
            <div className="flex flex-col items-center gap-10 w-full max-w-md animate-fade-in">
              <div className="text-center space-y-3">
                <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-b from-white via-zinc-200 to-zinc-500 bg-clip-text text-transparent leading-tight">
                  Detect any outfit
                </h2>
                <p className="text-zinc-500 text-[13px] max-w-xs mx-auto leading-relaxed">
                  Upload a photo and AI identifies each clothing item. Click any piece to find it online.
                </p>
              </div>

              <label
                className={`group relative flex flex-col items-center justify-center w-full h-72 border border-white/[0.06] rounded-2xl cursor-pointer hover:border-white/[0.12] bg-white/[0.02] hover:bg-white/[0.04] transition-all duration-500 ${dragging ? "border-white/[0.12] bg-white/[0.04]" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { setDragging(false); handleDrop(e); }}
              >
                <div className={`absolute inset-[-1px] rounded-2xl bg-gradient-to-br from-purple-500/0 via-transparent to-blue-500/0 group-hover:from-purple-500/20 group-hover:to-blue-500/20 transition-all duration-700 pointer-events-none ${dragging ? "from-purple-500/20 to-blue-500/20" : ""}`} />

                <div className="relative flex flex-col items-center gap-5">
                  <div className={`w-14 h-14 rounded-xl bg-white/[0.04] group-hover:bg-white/[0.08] border border-white/[0.06] flex items-center justify-center transition-all duration-300 group-hover:scale-105 ${dragging ? "bg-white/[0.08] scale-105" : ""}`}>
                    <svg className={`w-6 h-6 text-zinc-500 group-hover:text-zinc-300 transition-colors duration-300 ${dragging ? "text-zinc-300" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                  </div>
                  <div className="text-center">
                    <span className="text-zinc-300 text-sm font-medium block">Drop image here</span>
                    <span className="text-zinc-600 text-xs mt-1.5 block">or click to browse</span>
                  </div>
                </div>
                <input type="file" className="hidden" accept="image/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </label>

              <div className="flex flex-wrap justify-center gap-2">
                {["Detect clothing layers", "Color-aware search", "Shop similar items"].map((t) => (
                  <span key={t} className="text-[11px] text-zinc-500 bg-white/[0.03] border border-white/[0.06] rounded-full px-3 py-1">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-5 w-full animate-fade-in-scale">
              {/* Status */}
              {stage === "encoding" && (
                <div className="flex flex-col items-center gap-2 bg-white/[0.03] border border-white/[0.06] rounded-xl px-5 py-3 backdrop-blur-sm min-w-[260px]">
                  <div className="flex items-center justify-between w-full">
                    <span className="text-sm text-zinc-400">{statusText}</span>
                    <span className="text-xs text-zinc-500 tabular-nums">{progress}%</span>
                  </div>
                  <div className="w-full h-1 bg-white/[0.06] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all duration-200 ease-out"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {stage === "ready" && (
                <div className="flex items-center gap-2 text-[13px] text-zinc-500">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" />
                  {statusText}
                </div>
              )}

              {stage === "idle" && statusText && (
                <div className="text-sm text-red-400/80 bg-red-500/[0.06] border border-red-500/10 rounded-xl px-4 py-2.5">{statusText}</div>
              )}

              {/* Canvas stack */}
              <div className="relative inline-block rounded-2xl overflow-hidden shadow-2xl shadow-black/50 ring-1 ring-white/[0.06]">
                <canvas ref={canvasRef} className="block" />
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
                        ? "0 0 0 9999px rgba(0, 0, 0, 0.45)"
                        : "0 0 0 9999px rgba(0, 0, 0, 0)",
                      zIndex: 10,
                    }}
                  >
                    {/* Top-right corner */}
                    <svg
                      className="absolute focus-box-transition"
                      style={{ top: -10, right: -10, opacity: focusDimmed ? 1 : 0, zIndex: 12 }}
                      width="24" height="24" viewBox="0 0 24 24" fill="none"
                    >
                      <path d="M2 2A20 20 0 0 1 22 22" stroke="rgba(255,255,255,0.85)" strokeWidth="4.5" strokeLinecap="round" />
                    </svg>
                    {/* Bottom-left corner */}
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
          className={`border-l border-white/[0.04] bg-zinc-950/60 backdrop-blur-xl overflow-y-auto transition-all duration-500 ease-in-out ${
            selectedCategory ? "lg:w-[400px]" : "lg:w-0 lg:border-0 lg:overflow-hidden"
          }`}
        >
          {selectedCategory && (
            <div className="p-5 animate-fade-in">
              {/* Panel header */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-[13px] font-semibold text-zinc-200">
                    Similar {selectedCategory}
                  </h2>
                  {searchQuery && (
                    <p className="text-[11px] text-zinc-600 mt-0.5 truncate max-w-[260px]">
                      &ldquo;{searchQuery}&rdquo;
                    </p>
                  )}
                </div>
                <button
                  onClick={closePanel}
                  className="text-zinc-600 hover:text-zinc-300 transition-colors p-1"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Loading */}
              {searching && (
                <div className="flex flex-col items-center justify-center py-20 gap-3">
                  <div className="relative w-7 h-7">
                    <div className="absolute inset-0 rounded-full border-2 border-white/[0.06]" />
                    <div className="absolute inset-0 rounded-full border-2 border-t-purple-400 border-r-transparent animate-spin" />
                  </div>
                  <span className="text-xs text-zinc-500">Searching...</span>
                </div>
              )}

              {/* No results */}
              {!searching && products.length === 0 && searchQuery && (
                <div className="text-center py-16">
                  <p className="text-sm text-zinc-600">No products found</p>
                  <p className="text-xs text-zinc-700 mt-1">Check your SERPER_API_KEY</p>
                </div>
              )}

              {/* Product grid */}
              <div className="grid grid-cols-2 gap-2.5 stagger-children">
                {products.map((product, i) => (
                  <a
                    key={`${product.id}-${i}`}
                    href={product.link || "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group rounded-xl overflow-hidden border border-white/[0.04] hover:border-white/[0.1] bg-white/[0.02] hover:bg-white/[0.05] transition-all duration-200"
                  >
                    <div className="aspect-square bg-zinc-900 overflow-hidden">
                      {product.image && (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                          src={product.image}
                          alt={product.name}
                          className="w-full h-full object-cover group-hover:scale-[1.04] transition-transform duration-500 ease-out"
                          loading="lazy"
                        />
                      )}
                    </div>
                    <div className="p-2.5 space-y-1">
                      <p className="text-[10px] text-zinc-600 uppercase tracking-wider font-medium">{product.brand}</p>
                      <p className="text-[13px] text-zinc-300 leading-snug line-clamp-2">{product.name}</p>
                      {product.price && (
                        <p className="text-[13px] font-semibold text-white">{product.price}</p>
                      )}
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
