import { useRef, useEffect } from 'react';
import { useCamera, Camera } from './components/camera/Camera';
import { useDetector } from './components/ai/Detector';
import { drawAROverlay } from './components/ar/AROverlay';

function App() {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  
  const { videoRef, streamRef, isRunning, startWebcam, stopWebcam } = useCamera();
  const { session, detect } = useDetector();

  // Start detection loop when webcam starts
  useEffect(() => {
    if (!isRunning || !session || !videoRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const loopDetection = () => {
      if (!streamRef.current || !videoRef.current) return;
      detect(videoRef.current, canvasRef, drawAROverlay);
      animationFrameRef.current = requestAnimationFrame(loopDetection);
    };

    loopDetection();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
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
    <div className="bg-gray-900 text-gray-100 font-sans text-center pt-8 min-h-screen">
      <h1 className="text-3xl mb-6">🏙️ WTC Detector</h1>
      <Camera videoRef={videoRef} />
      <canvas ref={canvasRef} width="640" height="480" className="rounded-lg block mx-auto my-4" />

      <div className="mt-5">
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
