import { useRef, useEffect, useState } from 'react';
import { useCamera, Camera } from './components/camera/Camera';
import { useDetector, BUILDING_CLASSES } from './components/ai/Detector';
import { drawAROverlay } from './components/ar/AROverlay';

function App() {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const detectionFrameRef = useRef(null);
  const frameCountRef = useRef(0);
  const isDetectingRef = useRef(false);
  const [lastDetections, setLastDetections] = useState([]);

  // Detection rate: detect every N frames (2 = ~30 FPS, 3 = ~20 FPS, 5 = ~12 FPS)
  const DETECTION_INTERVAL = 3;

  const { videoRef, streamRef, isRunning, startWebcam, stopWebcam } = useCamera();
  const { session, detect } = useDetector();

  // Separate video rendering loop (runs at full speed for smooth video)
  useEffect(() => {
    if (!isRunning || !videoRef.current || !canvasRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const renderVideo = () => {
      if (!videoRef.current || !canvasRef.current) return;

      const ctx = canvasRef.current.getContext("2d");
      const video = videoRef.current;
      const iw = video.videoWidth || 640;
      const ih = video.videoHeight || 480;

      // Only resize if dimensions changed - use actual video dimensions
      if (canvasRef.current.width !== iw || canvasRef.current.height !== ih) {
        canvasRef.current.width = iw;
        canvasRef.current.height = ih;
        // Update CSS to maintain aspect ratio dynamically
        const aspectRatio = iw / ih;
        canvasRef.current.style.aspectRatio = `${aspectRatio}`;
      }

      // Clear and draw video frame
      ctx.clearRect(0, 0, iw, ih);
      ctx.drawImage(video, 0, 0, iw, ih);

      // Redraw detection boxes and AR overlay on top (so they persist)
      if (lastDetections.length > 0) {
        // Scale font and line width based on canvas size for mobile
        const scale = Math.min(iw / 640, ih / 480);
        const lineWidth = Math.max(2, 3 * scale);
        const fontSize = Math.max(14, 18 * scale);
        
        // Draw boxes
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        ctx.font = `${fontSize}px monospace`;

        for (const detection of lastDetections) {
          const { box, confidence, label } = detection;
          const { x1, y1, x2, y2 } = box;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          const labelText = `${label} ${(confidence * 100).toFixed(1)}%`;
          const textWidth = ctx.measureText(labelText).width;
          ctx.fillRect(x1, y1 - 20, textWidth + 8, 20);
          ctx.fillStyle = "white";
          ctx.fillText(labelText, x1 + 4, y1 - 4);
          ctx.fillStyle = "red";
        }

        // Draw AR overlay on top
        drawAROverlay(ctx, lastDetections.map(d => [d.box.x1, d.box.y1, d.box.x2, d.box.y2, d.confidence, d.classId]));
      }

      animationFrameRef.current = requestAnimationFrame(renderVideo);
    };

    renderVideo();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isRunning, videoRef, lastDetections]);

  // Detection loop (runs at lower frequency)
  useEffect(() => {
    if (!isRunning || !session || !videoRef.current) {
      isDetectingRef.current = false;
      frameCountRef.current = 0;
      if (detectionFrameRef.current) {
        cancelAnimationFrame(detectionFrameRef.current);
        detectionFrameRef.current = null;
      }
      return;
    }

    const runDetection = async () => {
      if (!streamRef.current || !videoRef.current || !canvasRef.current) return;

      frameCountRef.current++;

      // Skip frames - only detect every N frames
      if (frameCountRef.current % DETECTION_INTERVAL !== 0) {
        detectionFrameRef.current = requestAnimationFrame(runDetection);
        return;
      }

      // Prevent stacking detection requests - wait for current detection to finish
      if (isDetectingRef.current) {
        detectionFrameRef.current = requestAnimationFrame(runDetection);
        return;
      }

      isDetectingRef.current = true;

      try {
        // Run detection - don't draw, just get results
        const detections = await detect(videoRef.current, canvasRef, null);
        
        // Convert detections to format with labels - only show top 1 detection
        // Also filter by confidence (should already be filtered, but double-check)
        const formattedDetections = detections
          .filter(([x1, y1, x2, y2, conf, classId]) => conf >= 0.6) // Ensure 60%+ confidence
          .slice(0, 1) // Only keep top 1 detection
          .map(([x1, y1, x2, y2, conf, classId]) => ({
            box: { x1, y1, x2, y2 },
            confidence: conf,
            classId: classId,
            label: classId !== undefined ? (BUILDING_CLASSES[classId] || `Building ${classId}`) : 'Unknown'
          }));
        
        setLastDetections(formattedDetections);
      } catch (error) {
        console.error("Detection error:", error);
      } finally {
        isDetectingRef.current = false;
        // Continue loop after detection completes
        detectionFrameRef.current = requestAnimationFrame(runDetection);
      }
    };

    runDetection();

    return () => {
      isDetectingRef.current = false;
      frameCountRef.current = 0;
      if (detectionFrameRef.current) {
        cancelAnimationFrame(detectionFrameRef.current);
        detectionFrameRef.current = null;
      }
    };
  }, [isRunning, session, detect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <div className="bg-gray-900 text-gray-100 font-sans text-center pt-4 sm:pt-8 min-h-screen pb-4">
      <h1 className="text-2xl sm:text-3xl mb-4 sm:mb-6 px-4">🏙️ Building Detector</h1>
      <Camera videoRef={videoRef} />
      <div className="px-4 flex justify-center">
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          className="rounded-lg block my-2 sm:my-4 w-full sm:w-auto sm:max-w-2xl h-auto"
          style={{ maxWidth: '100%', maxHeight: '70vh', objectFit: 'contain' }}
        />
      </div>

      <div className="mt-4 sm:mt-5 px-4 flex flex-col sm:flex-row gap-3 sm:gap-0 justify-center items-center">
        <button
          id="startCam"
          onClick={startWebcam}
          disabled={!session || isRunning}
          className="w-full sm:w-auto sm:mx-2 px-6 sm:px-4 py-3 sm:py-2.5 text-lg sm:text-base rounded-md border-none bg-blue-500 text-white cursor-pointer active:bg-blue-600 hover:bg-blue-400 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors touch-manipulation"
        >
          Start Webcam
        </button>
        <button
          id="stopCam"
          onClick={stopWebcam}
          disabled={!isRunning}
          className="w-full sm:w-auto sm:mx-2 px-6 sm:px-4 py-3 sm:py-2.5 text-lg sm:text-base rounded-md border-none bg-blue-500 text-white cursor-pointer active:bg-blue-600 hover:bg-blue-400 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors touch-manipulation"
        >
          Stop Webcam
        </button>
      </div>
    </div>
  );
}

export default App;
