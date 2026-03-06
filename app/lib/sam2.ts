let ort: typeof import("onnxruntime-web/all");

async function getOrt() {
  if (!ort) {
    ort = await import("onnxruntime-web/all");
  }
  return ort;
}

const MODEL_BASE =
  "https://storage.googleapis.com/lb-artifacts-testing-public/sam2";

const ENCODER_URL = `${MODEL_BASE}/sam2_hiera_tiny.encoder.ort`;
const DECODER_URL = `${MODEL_BASE}/sam2_hiera_tiny.decoder.onnx`;

type OrtTensor = import("onnxruntime-web/all").Tensor;
type OrtSession = import("onnxruntime-web/all").InferenceSession;

async function getSession(url: string): Promise<OrtSession> {
  const ortModule = await getOrt();
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");

  for (const provider of providers) {
    try {
      const session = await ortModule.InferenceSession.create(url, {
        executionProviders: [provider],
      });
      console.log(`[SAM2] Session created with provider: ${provider}`);
      return session;
    } catch {
      console.warn(`[SAM2] Failed with ${provider}, trying next...`);
    }
  }
  throw new Error("Failed to create ONNX session with any provider");
}

export interface EncoderResult {
  imageEmbed: OrtTensor;
  highResFeats0: OrtTensor;
  highResFeats1: OrtTensor;
}

export interface Point {
  x: number;
  y: number;
  label: number;
}

let encoderSession: OrtSession | null = null;
let decoderSession: OrtSession | null = null;

export async function loadModels(
  onProgress?: (stage: string) => void
): Promise<void> {
  if (!encoderSession) {
    onProgress?.("Downloading encoder model...");
    encoderSession = await getSession(ENCODER_URL);
  }
  if (!decoderSession) {
    onProgress?.("Downloading decoder model...");
    decoderSession = await getSession(DECODER_URL);
  }
  onProgress?.("Models ready");
}

export async function imageToTensor(image: HTMLImageElement | HTMLCanvasElement): Promise<{
  tensor: OrtTensor;
  originalWidth: number;
  originalHeight: number;
}> {
  const ortModule = await getOrt();
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext("2d")!;

  let originalWidth: number;
  let originalHeight: number;

  if (image instanceof HTMLImageElement) {
    originalWidth = image.naturalWidth;
    originalHeight = image.naturalHeight;
  } else {
    originalWidth = image.width;
    originalHeight = image.height;
  }

  const maxDim = Math.max(originalWidth, originalHeight);
  const scale = 1024 / maxDim;
  const scaledW = originalWidth * scale;
  const scaledH = originalHeight * scale;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 1024, 1024);
  ctx.drawImage(image, 0, 0, scaledW, scaledH);

  const imageData = ctx.getImageData(0, 0, 1024, 1024).data;
  const inputArray = new Float32Array(3 * 1024 * 1024);

  for (let i = 0; i < 1024 * 1024; i++) {
    inputArray[i] = (imageData[i * 4] / 255.0) * 2 - 1;
    inputArray[i + 1024 * 1024] = (imageData[i * 4 + 1] / 255.0) * 2 - 1;
    inputArray[i + 2 * 1024 * 1024] = (imageData[i * 4 + 2] / 255.0) * 2 - 1;
  }

  return {
    tensor: new ortModule.Tensor("float32", inputArray, [1, 3, 1024, 1024]),
    originalWidth,
    originalHeight,
  };
}

export async function encodeImage(
  imageTensor: OrtTensor
): Promise<EncoderResult> {
  if (!encoderSession) throw new Error("Encoder not loaded");
  const results = await encoderSession.run({ image: imageTensor });
  return {
    imageEmbed: results["image_embed"],
    highResFeats0: results["high_res_feats_0"],
    highResFeats1: results["high_res_feats_1"],
  };
}

export async function decode(
  embeddings: EncoderResult,
  points: Point[]
): Promise<{ masks: OrtTensor; scores: OrtTensor }> {
  if (!decoderSession) throw new Error("Decoder not loaded");
  const ortModule = await getOrt();

  const numPoints = points.length;
  const pointCoordsData = new Float32Array(numPoints * 2);
  const pointLabelsData = new Float32Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    pointCoordsData[i * 2] = points[i].x;
    pointCoordsData[i * 2 + 1] = points[i].y;
    pointLabelsData[i] = points[i].label;
  }

  const feeds: Record<string, OrtTensor> = {
    image_embed: embeddings.imageEmbed,
    high_res_feats_0: embeddings.highResFeats0,
    high_res_feats_1: embeddings.highResFeats1,
    point_coords: new ortModule.Tensor("float32", pointCoordsData, [1, numPoints, 2]),
    point_labels: new ortModule.Tensor("float32", pointLabelsData, [1, numPoints]),
    mask_input: new ortModule.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]),
    has_mask_input: new ortModule.Tensor("float32", new Float32Array([0.0]), [1]),
  };

  const results = await decoderSession.run(feeds);
  return {
    masks: results["masks"],
    scores: results["iou_predictions"],
  };
}

export function maskToCanvas(
  maskTensor: OrtTensor,
  scoreTensor: OrtTensor,
  width: number,
  height: number,
  color: [number, number, number] = [59, 130, 246]
): ImageData {
  const maskData = maskTensor.data as Float32Array;
  const scores = scoreTensor.data as Float32Array;

  let bestIdx = 0;
  for (let i = 1; i < 3; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }

  const maskSize = 1024;
  const offset = bestIdx * maskSize * maskSize;
  const imageData = new ImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcX = Math.floor((x / width) * maskSize);
      const srcY = Math.floor((y / height) * maskSize);
      const val = maskData[offset + srcY * maskSize + srcX];

      const pixelIdx = (y * width + x) * 4;
      if (val > 0) {
        imageData.data[pixelIdx] = color[0];
        imageData.data[pixelIdx + 1] = color[1];
        imageData.data[pixelIdx + 2] = color[2];
        imageData.data[pixelIdx + 3] = 120;
      }
    }
  }

  return imageData;
}

export function getBestMaskBounds(
  maskTensor: OrtTensor,
  scoreTensor: OrtTensor,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number; w: number; h: number } | null {
  const maskData = maskTensor.data as Float32Array;
  const scores = scoreTensor.data as Float32Array;

  let bestIdx = 0;
  for (let i = 1; i < 3; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }

  const maskSize = 1024;
  const offset = bestIdx * maskSize * maskSize;

  let minX = maskSize, minY = maskSize, maxX = 0, maxY = 0;
  let found = false;

  for (let y = 0; y < maskSize; y++) {
    for (let x = 0; x < maskSize; x++) {
      if (maskData[offset + y * maskSize + x] > 0) {
        found = true;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (!found) return null;

  return {
    x: Math.floor((minX / maskSize) * canvasWidth),
    y: Math.floor((minY / maskSize) * canvasHeight),
    w: Math.ceil(((maxX - minX) / maskSize) * canvasWidth),
    h: Math.ceil(((maxY - minY) / maskSize) * canvasHeight),
  };
}
