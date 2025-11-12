import { useRef, useState } from 'react';

export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);

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
        };
      }
    } catch (err) {
      console.error("Error accessing webcam: ", err);
      alert("Failed to access webcam. Please check permissions and try again.");
    }
  };

  const stopWebcam = () => {
    setIsRunning(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  return {
    videoRef,
    streamRef,
    isRunning,
    startWebcam,
    stopWebcam,
  };
}

export function Camera({ videoRef }) {
  return (
    <video
      ref={videoRef}
      width="640"
      height="480"
      autoPlay
      muted
      playsInline
      className="hidden"
    />
  );
}

