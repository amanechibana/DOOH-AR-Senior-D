import { useRef, useState } from 'react';

export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);

  const startWebcam = async () => {
    if (isRunning || streamRef.current) return;
    try {
      // Request rear camera (back camera) on mobile devices
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          facingMode: 'environment' // 'environment' = rear camera, 'user' = front camera
        }
      });
      streamRef.current = stream;
      
      // Apply 3x zoom if supported (mobile devices)
      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack && videoTrack.getCapabilities) {
        const capabilities = videoTrack.getCapabilities();
        if (capabilities.zoom) {
          const maxZoom = capabilities.zoom.max || 1;
          const zoomLevel = Math.min(3.0, maxZoom); // 3x zoom, but don't exceed max
          try {
            await videoTrack.applyConstraints({
              advanced: [{ zoom: zoomLevel }]
            });
            console.log(`Applied ${zoomLevel}x zoom`);
          } catch (zoomErr) {
            console.warn("Could not apply zoom:", zoomErr);
          }
        }
      }
      
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

