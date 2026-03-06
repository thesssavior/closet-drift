"use client";

import { useState, useRef, useCallback } from "react";
import {
  encodeImage,
  decodeMask,
  searchProducts,
  type Product,
} from "./lib/sam2";

type Stage = "idle" | "encoding" | "ready";

const CATEGORY_COLORS: Record<number, [number, number, number]> = {
  1: [45, 90, 65],
  3: [200, 85, 60],
  4: [280, 75, 65],
  5: [330, 80, 65],
  6: [210, 80, 55],
  7: [350, 75, 60],
  8: [30, 70, 50],
  9: [160, 70, 50],
  10: [160, 70, 50],
  16: [140, 65, 55],
  17: [15, 80, 60],
};

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [statusText, setStatusText] = useState("");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [hoveredCategory, setHoveredCategory] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const shimmerRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const hashRef = useRef("");
  const fileRef = useRef<File | null>(null);
  const dimsRef = useRef({ scale: 1, origW: 0, origH: 0 });
  const hoverThrottleRef = useRef(0);
  const lastCategoryRef = useRef(-1);
  const shimmerAnimRef = useRef(0);
  const maskImageDataRef = useRef<ImageData | null>(null);

  const drawImage = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current!;
    const overlay = overlayRef.current!;
    const ctx = canvas.getContext("2d")!;

    const container = containerRef.current ?? canvas.parentElement!;
    const maxW = Math.min(container.clientWidth - 48, 860);
    const maxH = Math.min(container.clientHeight - 80, 640);
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
    const w = Math.floor(img.naturalWidth * scale);
    const h = Math.floor(img.naturalHeight * scale);

    canvas.width = w;
    canvas.height = h;
    overlay.width = w;
    overlay.height = h;
    if (shimmerRef.current) {
      shimmerRef.current.width = w;
      shimmerRef.current.height = h;
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

          // Simulated progress that slows as it approaches 90%
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

  // --- Shimmer animation ---
  const stopShimmer = useCallback(() => {
    if (shimmerAnimRef.current) {
      cancelAnimationFrame(shimmerAnimRef.current);
      shimmerAnimRef.current = 0;
    }
    if (shimmerRef.current) {
      const ctx = shimmerRef.current.getContext("2d")!;
      ctx.clearRect(0, 0, shimmerRef.current.width, shimmerRef.current.height);
    }
  }, []);

  const startShimmer = useCallback(
    (maskData: ImageData, hue: number) => {
      stopShimmer();
      const shimmer = shimmerRef.current;
      if (!shimmer) return;
      const ctx = shimmer.getContext("2d")!;
      const w = shimmer.width;
      const h = shimmer.height;
      const startTime = performance.now();

      const animate = (now: number) => {
        const elapsed = now - startTime;
        const t = (elapsed % 2200) / 2200;
        const pos = -0.3 + t * 1.6;

        ctx.clearRect(0, 0, w, h);

        const cos = Math.cos(Math.PI / 4);
        const sin = Math.sin(Math.PI / 4);
        const diag = w * cos + h * sin;
        const center = pos * diag;
        const spread = diag * 0.12;

        const grad = ctx.createLinearGradient(-h * sin, 0, w * cos, h * sin + w * cos);
        const p = center / diag;
        const cl = (v: number) => Math.max(0, Math.min(1, v));

        grad.addColorStop(cl(p - spread / diag - 0.01), "transparent");
        grad.addColorStop(cl(p - spread / diag), `hsla(${hue}, 40%, 88%, 0.25)`);
        grad.addColorStop(cl(p), `hsla(${hue}, 30%, 96%, 0.4)`);
        grad.addColorStop(cl(p + spread / diag), `hsla(${hue}, 40%, 88%, 0.25)`);
        grad.addColorStop(cl(p + spread / diag + 0.01), "transparent");

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Clip to mask
        const shimmerData = ctx.getImageData(0, 0, w, h);
        for (let i = 0; i < maskData.data.length; i += 4) {
          if (maskData.data[i] <= 20) shimmerData.data[i + 3] = 0;
        }
        ctx.putImageData(shimmerData, 0, 0);

        shimmerAnimRef.current = requestAnimationFrame(animate);
      };
      shimmerAnimRef.current = requestAnimationFrame(animate);
    },
    [stopShimmer]
  );

  // --- Mask rendering ---
  const renderMask = useCallback(
    (maskB64: string, categoryId: number, mode: "hover" | "selected") => {
      const overlay = overlayRef.current!;
      const ctx = overlay.getContext("2d")!;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      const [h, s, l] = CATEGORY_COLORS[categoryId] ?? [220, 70, 60];

      const maskImg = new window.Image();
      maskImg.onload = () => {
        const tmp = document.createElement("canvas");
        tmp.width = overlay.width;
        tmp.height = overlay.height;
        const tmpCtx = tmp.getContext("2d")!;
        tmpCtx.drawImage(maskImg, 0, 0, overlay.width, overlay.height);
        const maskData = tmpCtx.getImageData(0, 0, overlay.width, overlay.height);
        maskImageDataRef.current = maskData;

        const out = ctx.createImageData(overlay.width, overlay.height);
        const rgb = hslToRgb(h, s, l);

        for (let i = 0; i < maskData.data.length; i += 4) {
          const a = maskData.data[i];
          if (a > 20) {
            const n = Math.min(a / 255, 1);
            const op = mode === "selected" ? n * 0.38 : n * 0.22;
            out.data[i] = rgb[0];
            out.data[i + 1] = rgb[1];
            out.data[i + 2] = rgb[2];
            out.data[i + 3] = Math.floor(op * 255);
          }
        }
        ctx.putImageData(out, 0, 0);

        // Edge glow
        ctx.globalCompositeOperation = "source-over";
        ctx.filter = "blur(4px)";
        ctx.globalAlpha = mode === "selected" ? 0.35 : 0.2;
        ctx.drawImage(overlay, 0, 0);
        ctx.filter = "none";
        ctx.globalAlpha = 1;

        startShimmer(maskData, h);
      };
      maskImg.src = `data:image/png;base64,${maskB64}`;
    },
    [startShimmer]
  );

  const clearOverlay = useCallback(() => {
    stopShimmer();
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d")!;
    ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
  }, [stopShimmer]);

  const handleMouseLeave = useCallback(() => {
    clearOverlay();
    setHoveredCategory("");
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
      const { x, y } = canvasToModelCoords(e.clientX - rect.left, e.clientY - rect.top);
      const canvas = canvasRef.current!;

      try {
        const result = await decodeMask(hashRef.current, x, y, canvas.width, canvas.height);
        if (result.mask) {
          if (result.categoryId === lastCategoryRef.current) return;
          lastCategoryRef.current = result.categoryId;
          setHoveredCategory(result.category);
          renderMask(result.mask, result.categoryId, "hover");
        } else {
          lastCategoryRef.current = -1;
          setHoveredCategory("");
          clearOverlay();
        }
      } catch {}
    },
    [stage, canvasToModelCoords, renderMask, clearOverlay]
  );

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLElement>) => {
      if (stage !== "ready" || !hashRef.current) return;
      const rect = canvasRef.current!.getBoundingClientRect();
      const { x, y } = canvasToModelCoords(e.clientX - rect.left, e.clientY - rect.top);
      const canvas = canvasRef.current!;

      try {
        const result = await decodeMask(hashRef.current, x, y, canvas.width, canvas.height);
        if (!result.mask) return;
        renderMask(result.mask, result.categoryId, "selected");
        setSelectedCategory(result.category);

        setSearching(true);
        setProducts([]);
        setSearchQuery("");
        const sr = await searchProducts(hashRef.current, result.categoryId);
        setSearchQuery(sr.query);
        setProducts(sr.products);
        setSearching(false);
      } catch {
        setSearching(false);
      }
    },
    [stage, canvasToModelCoords, renderMask]
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
    hashRef.current = "";
  };

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
                {/* Hover gradient border */}
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
                {hoveredCategory && (
                  <div className="absolute -top-11 left-1/2 z-20 pointer-events-none animate-float-label">
                    <div className="bg-zinc-900/95 backdrop-blur-xl border border-white/[0.1] text-zinc-100 text-xs font-medium px-4 py-2 rounded-full shadow-xl shadow-black/40 whitespace-nowrap">
                      {hoveredCategory}
                    </div>
                  </div>
                )}
                <canvas ref={canvasRef} className="block" />
                <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                <canvas ref={shimmerRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                <div
                  className="absolute inset-0"
                  style={{ cursor: stage === "ready" ? "pointer" : "default" }}
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
                  onClick={() => { setProducts([]); setSearchQuery(""); setSelectedCategory(""); handleMouseLeave(); }}
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

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}
