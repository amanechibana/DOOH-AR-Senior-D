import { BUILDING_CLASSES } from '../ai/Detector';

// AR Overlay function
export const drawAROverlay = (ctx, boxes) => {
  if (boxes.length === 0) return;

  const [x1, y1, x2, y2, conf, classId] = boxes[0];
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  
  // Get building name from class ID
  const buildingName = BUILDING_CLASSES[classId] || `Building ${classId}`;

  // Scale elements based on canvas size for mobile responsiveness
  const canvas = ctx.canvas;
  const scale = Math.min(canvas.width / 640, canvas.height / 480);
  const baseRadius = 30 * scale;
  const baseCrosshair = 40 * scale;
  const baseBracket = 30 * scale;
  const baseInfoHeight = 80 * scale;
  const baseInfoWidth = Math.max(300 * scale, buildingName.length * 10 * scale + 40 * scale);

  ctx.save();

  // Pulsing green circle at center
  const time = Date.now() / 1000;
  const pulse = Math.sin(time * 2) * 0.5 + 0.5;
  const radius = baseRadius + pulse * (20 * scale);

  ctx.fillStyle = `rgba(0, 255, 0, ${0.3 + pulse * 0.3})`;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fill();

  // Crosshair at center
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = Math.max(2, 3 * scale);
  ctx.beginPath();
  ctx.moveTo(centerX - baseCrosshair, centerY);
  ctx.lineTo(centerX + baseCrosshair, centerY);
  ctx.moveTo(centerX, centerY - baseCrosshair);
  ctx.lineTo(centerX, centerY + baseCrosshair);
  ctx.stroke();

  // Info box above building
  const infoY = y1 - (100 * scale);
  const infoWidth = baseInfoWidth;
  const infoHeight = baseInfoHeight;
  const infoX = centerX - infoWidth / 2;

  ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
  ctx.fillRect(infoX, infoY, infoWidth, infoHeight);

  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = Math.max(1, 2 * scale);
  ctx.strokeRect(infoX, infoY, infoWidth, infoHeight);

  ctx.fillStyle = "#0f0";
  ctx.font = `bold ${Math.max(14, 20 * scale)}px Arial`;
  ctx.textAlign = "center";
  ctx.fillText(buildingName, centerX, infoY + (28 * scale));
  ctx.font = `${Math.max(12, 16 * scale)}px Arial`;
  ctx.fillText(`Confidence: ${(conf * 100).toFixed(1)}%`, centerX, infoY + (52 * scale));

  // Corner brackets around building
  const bracketSize = baseBracket;
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = Math.max(2, 4 * scale);

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

