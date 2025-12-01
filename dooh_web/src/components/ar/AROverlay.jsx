// AR Overlay function
export const drawAROverlay = (ctx, boxes) => {
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
  ctx.fillText("🏢", centerX, infoY + 28);
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

