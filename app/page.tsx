"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  loadModels,
  imageToTensor,
  encodeImage,
  decode,
  maskToCanvas,
  getBestMaskBounds,
  type EncoderResult,
  type Point,
} from "./lib/sam2";
import { getMockProducts, hashPoint, type Product } from "./lib/products";

type Stage = "idle" | "loading-models" | "encoding" | "ready";

export default function Home() {
  const [stage, setStage] = useState<Stage>("idle");
  const [statusText, setStatusText] = useState("");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const embeddingsRef = useRef<EncoderResult | null>(null);
  const imgDimsRef = useRef<{ w: number; h: number; scale: number; offsetX: number; offsetY: number }>({
    w: 0, h: 0, scale: 1, offsetX: 0, offsetY: 0,
  });
  const lastMaskRef = useRef<{ masks: any; scores: any } | null>(null);
  const hoverThrottleRef = useRef<number>(0);

  const drawImage = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current!;
    const overlay = overlayRef.current!;
    const ctx = canvas.getContext("2d")!;

    const maxW = canvas.parentElement!.clientWidth;
    const maxH = window.innerHeight * 0.75;
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
    const w = Math.floor(img.naturalWidth * scale);
    const h = Math.floor(img.naturalHeight * scale);

    canvas.width = w;
    canvas.height = h;
    overlay.width = w;
    overlay.height = h;
    ctx.drawImage(img, 0, 0, w, h);

    // Compute mapping from canvas coords to 1024x1024 model coords
    const maxDim = Math.max(img.naturalWidth, img.naturalHeight);
    const modelScale = 1024 / maxDim;
    imgDimsRef.current = {
      w, h,
      scale: scale,
      offsetX: 0,
      offsetY: 0,
    };
  }, []);

  const canvasToModelCoords = useCallback(
    (canvasX: number, canvasY: number) => {
      const img = imageRef.current!;
      const dims = imgDimsRef.current;
      // Canvas coords -> original image coords -> model (1024x1024) coords
      const origX = canvasX / dims.scale;
      const origY = canvasY / dims.scale;
      const maxDim = Math.max(img.naturalWidth, img.naturalHeight);
      const modelScale = 1024 / maxDim;
      return {
        x: origX * modelScale,
        y: origY * modelScale,
      };
    },
    []
  );

  const handleFile = useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file);
      setImageSrc(url);
      setProducts([]);
      setSelectedProduct(null);
      lastMaskRef.current = null;

      const img = new Image();
      img.onload = async () => {
        imageRef.current = img;
        drawImage(img);

        try {
          setStage("loading-models");
          setStatusText("Loading AI models (first time may take a minute)...");
          await loadModels((s) => setStatusText(s));

          setStage("encoding");
          setStatusText("Analyzing image...");
          const { tensor } = await imageToTensor(img);
          const embeddings = await encodeImage(tensor);
          embeddingsRef.current = embeddings;

          setStage("ready");
          setStatusText("Hover over clothing items to highlight them");
        } catch (err: any) {
          console.error(err);
          setStage("idle");
          setStatusText(`Error: ${err.message}`);
        }
      };
      img.src = url;
    },
    [drawImage]
  );

  const handleHover = useCallback(
    async (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (stage !== "ready" || !embeddingsRef.current) return;

      const now = Date.now();
      if (now - hoverThrottleRef.current < 100) return;
      hoverThrottleRef.current = now;

      const rect = canvasRef.current!.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;
      const { x, y } = canvasToModelCoords(canvasX, canvasY);

      const point: Point = { x, y, label: 1 };

      try {
        const result = await decode(embeddingsRef.current, [point]);
        lastMaskRef.current = result;

        const overlay = overlayRef.current!;
        const ctx = overlay.getContext("2d")!;
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const maskImageData = maskToCanvas(
          result.masks,
          result.scores,
          overlay.width,
          overlay.height,
          [59, 130, 246]
        );
        ctx.putImageData(maskImageData, 0, 0);
      } catch (err) {
        // Decoder may fail on edge coordinates, ignore
      }
    },
    [stage, canvasToModelCoords]
  );

  const handleMouseLeave = useCallback(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d")!;
    ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
  }, []);

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (stage !== "ready" || !embeddingsRef.current) return;

      const rect = canvasRef.current!.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;
      const { x, y } = canvasToModelCoords(canvasX, canvasY);

      // Run decode at clicked point for a clean mask
      const point: Point = { x, y, label: 1 };
      const result = await decode(embeddingsRef.current, [point]);
      lastMaskRef.current = result;

      // Draw the mask with a "selected" color
      const overlay = overlayRef.current!;
      const ctx = overlay.getContext("2d")!;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      const maskImageData = maskToCanvas(
        result.masks,
        result.scores,
        overlay.width,
        overlay.height,
        [139, 92, 246] // purple for selected
      );
      ctx.putImageData(maskImageData, 0, 0);

      // Get bounding box and generate mock products
      const bounds = getBestMaskBounds(
        result.masks,
        result.scores,
        overlay.width,
        overlay.height
      );

      const seed = hashPoint(x, y);
      const mockProducts = getMockProducts(seed);
      setProducts(mockProducts);
      setSelectedProduct(null);
    },
    [stage, canvasToModelCoords]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">
          Fit Detector
        </h1>
        <p className="text-sm text-zinc-400 mt-1">
          Upload a fit pic. Hover to detect items. Click to search similar products.
        </p>
      </header>

      <div className="flex flex-col lg:flex-row h-[calc(100vh-80px)]">
        {/* Left: Image area */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 overflow-auto">
          {!imageSrc ? (
            <label
              className="flex flex-col items-center justify-center w-full max-w-xl h-96 border-2 border-dashed border-zinc-700 rounded-2xl cursor-pointer hover:border-zinc-500 hover:bg-zinc-900/50 transition-all"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              <svg
                className="w-12 h-12 text-zinc-600 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 16v-8m0 0l-3 3m3-3l3 3M6.75 20.25h10.5A2.25 2.25 0 0019.5 18V6a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6v12a2.25 2.25 0 002.25 2.25z"
                />
              </svg>
              <span className="text-zinc-400 text-lg font-medium">
                Drop a fit pic here
              </span>
              <span className="text-zinc-500 text-sm mt-1">
                or click to browse
              </span>
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
            </label>
          ) : (
            <div className="flex flex-col items-center gap-4 w-full">
              {/* Status bar */}
              {stage !== "ready" && stage !== "idle" && (
                <div className="flex items-center gap-3 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 w-full max-w-xl">
                  <div className="relative w-5 h-5">
                    <div className="absolute inset-0 rounded-full border-2 border-zinc-700" />
                    <div className="absolute inset-0 rounded-full border-2 border-t-blue-500 animate-spin" />
                  </div>
                  <span className="text-sm text-zinc-300">{statusText}</span>
                </div>
              )}

              {stage === "ready" && (
                <div className="flex items-center gap-2 text-sm text-zinc-400">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  {statusText}
                </div>
              )}

              {/* Canvas container */}
              <div className="relative inline-block">
                <canvas ref={canvasRef} className="rounded-xl" />
                <canvas
                  ref={overlayRef}
                  className="absolute top-0 left-0 rounded-xl"
                  style={{ cursor: stage === "ready" ? "crosshair" : "default" }}
                  onMouseMove={handleHover}
                  onMouseLeave={handleMouseLeave}
                  onClick={handleClick}
                />
              </div>

              {/* Reset button */}
              <button
                onClick={() => {
                  setImageSrc(null);
                  setStage("idle");
                  setProducts([]);
                  setSelectedProduct(null);
                  embeddingsRef.current = null;
                  lastMaskRef.current = null;
                }}
                className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                Upload different image
              </button>
            </div>
          )}
        </div>

        {/* Right: Product gallery */}
        <div
          className={`lg:w-96 border-l border-zinc-800 bg-zinc-900/50 overflow-y-auto transition-all ${
            products.length === 0 ? "lg:w-0 lg:border-0 lg:overflow-hidden" : ""
          }`}
        >
          {products.length > 0 && (
            <div className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">
                  Similar Products
                </h2>
                <button
                  onClick={() => {
                    setProducts([]);
                    setSelectedProduct(null);
                    handleMouseLeave();
                  }}
                  className="text-xs text-zinc-500 hover:text-zinc-300"
                >
                  Close
                </button>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {products.map((product) => (
                  <button
                    key={product.id}
                    onClick={() => setSelectedProduct(product)}
                    className={`group text-left rounded-xl overflow-hidden border transition-all ${
                      selectedProduct?.id === product.id
                        ? "border-purple-500 bg-zinc-800"
                        : "border-zinc-800 hover:border-zinc-600 bg-zinc-900"
                    }`}
                  >
                    <div className="aspect-[3/4] bg-zinc-800 overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={product.image}
                        alt={product.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        loading="lazy"
                      />
                    </div>
                    <div className="p-2.5">
                      <p className="text-xs text-zinc-500">{product.brand}</p>
                      <p className="text-sm text-zinc-200 leading-tight mt-0.5 line-clamp-2">
                        {product.name}
                      </p>
                      <p className="text-sm font-semibold text-zinc-100 mt-1">
                        {product.price}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
