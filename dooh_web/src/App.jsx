import { useState, useEffect, useRef } from 'react'

// ONNX Runtime will be loaded dynamically from CDN
let ort = null;

// MODEL CONSTANTS (Based on YOLOv8 Segmentation Output)
const NUM_FEATURES = 37; // 4 (box) + 1 (score) + 32 (mask coeffs)
const NUM_BOX_SCORE = 5; // The first 5 features contain the box and score.

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Load model on mount
  useEffect(() => {
    const loadModel = async () => {
      // Dynamically import ONNX Runtime from CDN
      if (!ort) {
        try {
          const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/esm/ort.min.js');
          ort = ortModule;
        } catch (error) {
          console.error("Failed to load ONNX Runtime:", error);
          return;
        }
      }

      console.log("⏳ Loading YOLO ONNX model...");

      // Set WASM file path to CDN location
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';
      ort.env.wasm.numThreads = 1;

      try {
        // Model file is in public/ folder, served from root in Vite
        const newSession = await ort.InferenceSession.create("/wtc_finetuned_32.onnx", {
          executionProviders: ["wasm"],
        });
        console.log("✅ Model loaded:", newSession.inputNames, "→", newSession.outputNames);
        setSession(newSession);
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    };

    loadModel();
  }, []);

  // Helper function: letterbox
  const letterbox = (img, newShape = 640) => {
    const canvasTmp = document.createElement("canvas");
    const ctxTmp = canvasTmp.getContext("2d");
    canvasTmp.width = newShape;
    canvasTmp.height = newShape;
    ctxTmp.fillStyle = "rgb(114,114,114)";
    ctxTmp.fillRect(0, 0, newShape, newShape);

    const ratio = Math.min(newShape / img.width, newShape / img.height);
    const newW = img.width * ratio;
    const newH = img.height * ratio;
    const dx = (newShape - newW) / 2;
    const dy = (newShape - newH) / 2;
    ctxTmp.drawImage(img, dx, dy, newW, newH);
    const imageData = ctxTmp.getImageData(0, 0, newShape, newShape);
    return { data: imageData.data, ratio, dx, dy };
  };

  // Helper function: NMS
  const nms = (boxes, iouThresh = 0.45) => {
    boxes.sort((a, b) => b[4] - a[4]);
    const keep = [];

    for (let i = 0; i < boxes.length; i++) {
      const [x1, y1, x2, y2, conf] = boxes[i];
      let shouldKeep = true;
      for (const kept of keep) {
        const [kx1, ky1, kx2, ky2] = kept;

        const interX1 = Math.max(x1, kx1);
        const interY1 = Math.max(y1, ky1);
        const interX2 = Math.min(x2, kx2);
        const interY2 = Math.min(y2, ky2);

        const inter = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);

        const area1 = (x2 - x1) * (y2 - y1);
        const area2 = (kx2 - kx1) * (ky2 - ky1);
        const union = area1 + area2 - inter;

        if (inter / union > iouThresh) {
          shouldKeep = false;
          break;
        }
      }
      if (shouldKeep) keep.push(boxes[i]);
    }
    return keep;
  };

  // Helper function: Load and scale image
  const loadAndScaleImage = (imageSource, maxDimension = 2000) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;

        if (width > maxDimension || height > maxDimension) {
          const ratio = Math.min(maxDimension / width, maxDimension / height);
          width *= ratio;
          height *= ratio;
          console.log(`Image scaled down from original size to ${Math.round(width)}x${Math.round(height)}`);
        }

        const canvasTmp = document.createElement("canvas");
        canvasTmp.width = width;
        canvasTmp.height = height;
        const ctxTmp = canvasTmp.getContext("2d");
        ctxTmp.drawImage(img, 0, 0, width, height);

        const scaledImg = new Image();
        scaledImg.onload = () => resolve(scaledImg);
        scaledImg.onerror = reject;
        scaledImg.src = canvasTmp.toDataURL();
      };
      img.onerror = reject;
      img.src = imageSource;
    });
  };

  // Detection function
  const detect = async (imageElement) => {
    if (!session || !canvasRef.current) {
      console.warn("Model not loaded yet or canvas not available");
      return;
    }

    const ctx = canvasRef.current.getContext("2d");
    const iw = imageElement.videoWidth || imageElement.width;
    const ih = imageElement.videoHeight || imageElement.height;
    canvasRef.current.width = iw;
    canvasRef.current.height = ih;
    ctx.drawImage(imageElement, 0, 0, iw, ih);

    // Preprocess
    const { data, ratio, dx, dy } = letterbox(imageElement);
    const w = 640;
    const h = 640;
    const img = new Float32Array(1 * 3 * h * w);

    // Convert pixel data (RGBA) to Float32Array (NCHW, normalized)
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const base = j;
      img[base] = data[i] / 255.0; // R
      img[w * h + base] = data[i + 1] / 255.0; // G
      img[2 * w * h + base] = data[i + 2] / 255.0; // B
    }

    const input = new ort.Tensor("float32", img, [1, 3, h, w]);

    // Run Inference
    const outputs = await session.run({ [session.inputNames[0]]: input });
    const output = outputs[session.outputNames[0]];
    const dataArr = output.data;
    const shape = output.dims;

    // YOLOv8 Decoding
    const numFeatures = shape[1];
    const numPred = shape[2];
    const confThresh = 0.2;
    const boxes = [];

    if (numFeatures !== NUM_FEATURES) {
      console.error(`Expected ${NUM_FEATURES} features but got ${numFeatures}. Check your model export!`);
      return;
    }

    // Iterate over predictions
    for (let i = 0; i < numPred; i++) {
      const x = dataArr[0 * numPred + i];
      const y = dataArr[1 * numPred + i];
      const wBox = dataArr[2 * numPred + i];
      const hBox = dataArr[3 * numPred + i];
      const conf = dataArr[4 * numPred + i];

      if (conf > confThresh) {
        let x1 = x - wBox / 2;
        let y1 = y - hBox / 2;
        let x2 = x + wBox / 2;
        let y2 = y + hBox / 2;

        // Rescale to original image space
        x1 = (x1 - dx) / ratio;
        y1 = (y1 - dy) / ratio;
        x2 = (x2 - dx) / ratio;
        y2 = (y2 - dy) / ratio;

        // Clamp to image bounds
        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        x2 = Math.min(iw, x2);
        y2 = Math.min(ih, y2);

        boxes.push([x1, y1, x2, y2, conf, 0]);
      }
    }

    const filtered = nms(boxes);
    console.log(`✅ Detections after NMS: ${filtered.length}`);

    // Draw boxes
    ctx.lineWidth = 3;
    ctx.strokeStyle = "red";
    ctx.fillStyle = "red";
    ctx.font = "18px monospace";

    for (const [x1, y1, x2, y2, conf] of filtered) {
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `WTC ${(conf * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x1, y1 - 20, textWidth + 8, 20);
      ctx.fillStyle = "white";
      ctx.fillText(label, x1 + 4, y1 - 4);
      ctx.fillStyle = "red";
    }

    drawAROverlay(ctx, filtered);
  };

  // AR Overlay function
  const drawAROverlay = (ctx, boxes) => {
    if (boxes.length === 0) return;

    const [x1, y1, x2, y2, conf] = boxes[0];
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;

    ctx.save();

    // Pulsing green circle at center
    const time = Date.now() / 1000;
    const pulse = Math.sin(time * 2) * 0.5 + 0.5;
    const radius = 30 + pulse * 20;

    ctx.fillStyle = `rgba(0, 255, 0, ${0.3 + pulse * 0.3})`;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();

    // Crosshair at center
    ctx.strokeStyle = "#0f0";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX - 40, centerY);
    ctx.lineTo(centerX + 40, centerY);
    ctx.moveTo(centerX, centerY - 40);
    ctx.lineTo(centerX, centerY + 40);
    ctx.stroke();

    // Info box above building
    const infoY = y1 - 80;
    const infoWidth = 300;
    const infoHeight = 60;
    const infoX = centerX - infoWidth / 2;

    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(infoX, infoY, infoWidth, infoHeight);

    ctx.strokeStyle = "#0f0";
    ctx.lineWidth = 2;
    ctx.strokeRect(infoX, infoY, infoWidth, infoHeight);

    ctx.fillStyle = "#0f0";
    ctx.font = "bold 24px Arial";
    ctx.textAlign = "center";
    ctx.fillText("🏢 World Trade Center", centerX, infoY + 28);
    ctx.font = "16px Arial";
    ctx.fillText(`Confidence: ${(conf * 100).toFixed(1)}%`, centerX, infoY + 48);

    // Corner brackets around building
    const bracketSize = 30;
    ctx.strokeStyle = "#0f0";
    ctx.lineWidth = 4;

    // Top-left bracket
    ctx.beginPath();
    ctx.moveTo(x1 + bracketSize, y1);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1, y1 + bracketSize);
    ctx.stroke();

    // Top-right bracket
    ctx.beginPath();
    ctx.moveTo(x2 - bracketSize, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + bracketSize);
    ctx.stroke();

    // Bottom-left bracket
    ctx.beginPath();
    ctx.moveTo(x1, y2 - bracketSize);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + bracketSize, y2);
    ctx.stroke();

    // Bottom-right bracket
    ctx.beginPath();
    ctx.moveTo(x2 - bracketSize, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - bracketSize);
    ctx.stroke();

    ctx.restore();
  };

  // Webcam functions
  const startWebcam = async () => {
    if (isRunning || streamRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsRunning(true);
          videoRef.current?.play();
          // Start detection loop after video is ready
          loopDetection();
        };
      }
    } catch (err) {
      console.error("Error accessing webcam: ", err);
      alert("Failed to access webcam. Please check permissions and try again.");
    }
  };

  const stopWebcam = () => {
    setIsRunning(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const loopDetection = () => {
    if (!streamRef.current || !videoRef.current) return;
    detect(videoRef.current);
    animationFrameRef.current = requestAnimationFrame(loopDetection);
  };

  // Image upload handler
  const handleImageUpload = async (e) => {
    stopWebcam();
    const file = e.target.files[0];
    if (!file) return;

    try {
      const scaledImage = await loadAndScaleImage(URL.createObjectURL(file), 2000);
      await detect(scaledImage);
    } catch (error) {
      console.error("Failed to load or scale image:", error);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="bg-gray-900 text-gray-100 font-sans text-center pt-8">
      <h1 className="text-3xl mb-6">🏙️ WTC Detector</h1>
      <video
        ref={videoRef}
        width="640"
        height="480"
        autoPlay
        muted
        playsInline
        className={`rounded-lg mx-auto my-4 ${isRunning ? 'block' : 'hidden'}`}
      />
      <canvas ref={canvasRef} width="640" height="480" className="rounded-lg block mx-auto my-4" />

      <div className="mt-5">
        <input
          type="file"
          id="upload"
          accept="image/*"
          onChange={handleImageUpload}
          disabled={!session}
          className="mx-2 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          id="startCam"
          onClick={startWebcam}
          disabled={!session || isRunning}
          className="mx-2 px-4 py-2.5 rounded-md border-none bg-blue-500 text-white cursor-pointer hover:bg-blue-400 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
        >
          Start Webcam
        </button>
        <button
          id="stopCam"
          onClick={stopWebcam}
          disabled={!isRunning}
          className="mx-2 px-4 py-2.5 rounded-md border-none bg-blue-500 text-white cursor-pointer hover:bg-blue-400 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
        >
          Stop Webcam
        </button>
      </div>
    </div>
  );
}

export default App;
