import React from 'react';

import styles from './NeuralNetworkBackground.module.css';

const FRAME_INTERVAL = 1000 / 48;
const DEVICE_PIXEL_RATIO_CAP = 1.5;
const BASE_PADDING = 96;

const ACCENT_COLORS = [
  [67, 240, 191],
  [88, 228, 214],
  [54, 196, 167],
];

const RENDER_MODES = {
  small: {
    minNodes: 30,
    maxNodes: 48,
    density: 14000,
    drift: 0.06,
    minRadius: 1.5,
    maxRadius: 2.5,
    minAlpha: 0.24,
    maxAlpha: 0.38,
    connectionDistance: 170,
    lineOpacity: 0.26,
    lineWidth: 0.92,
    glowBlur: 13,
    glowBoost: 0.14,
  },
  compact: {
    minNodes: 48,
    maxNodes: 76,
    density: 17000,
    drift: 0.09,
    minRadius: 1.75,
    maxRadius: 2.9,
    minAlpha: 0.26,
    maxAlpha: 0.42,
    connectionDistance: 200,
    lineOpacity: 0.32,
    lineWidth: 1.08,
    glowBlur: 17,
    glowBoost: 0.16,
  },
  full: {
    minNodes: 72,
    maxNodes: 108,
    density: 18500,
    drift: 0.125,
    minRadius: 2,
    maxRadius: 3.6,
    minAlpha: 0.3,
    maxAlpha: 0.52,
    connectionDistance: 232,
    lineOpacity: 0.38,
    lineWidth: 1.22,
    glowBlur: 20,
    glowBoost: 0.18,
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function getRenderMode(width) {
  if (width < 640) {
    return RENDER_MODES.small;
  }

  if (width < 980) {
    return RENDER_MODES.compact;
  }

  return RENDER_MODES.full;
}

function blendColor(colorA, colorB) {
  return [
    Math.round((colorA[0] + colorB[0]) / 2),
    Math.round((colorA[1] + colorB[1]) / 2),
    Math.round((colorA[2] + colorB[2]) / 2),
  ];
}

function createNode(width, height, mode) {
  const angle = Math.random() * Math.PI * 2;
  const speed = mode.drift;
  const color = ACCENT_COLORS[Math.floor(Math.random() * ACCENT_COLORS.length)];

  return {
    x: randomBetween(-BASE_PADDING, width + BASE_PADDING),
    y: randomBetween(-BASE_PADDING, height + BASE_PADDING),
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    radius: randomBetween(mode.minRadius, mode.maxRadius),
    alpha: randomBetween(mode.minAlpha, mode.maxAlpha),
    color,
  };
}

function createNodes(width, height, mode) {
  const nodeCount = clamp(
    Math.round((width * height) / mode.density),
    mode.minNodes,
    mode.maxNodes,
  );

  return Array.from({length: nodeCount}, () => createNode(width, height, mode));
}

function wrapNode(node, width, height) {
  if (node.x < -BASE_PADDING) {
    node.x = width + BASE_PADDING;
  } else if (node.x > width + BASE_PADDING) {
    node.x = -BASE_PADDING;
  }

  if (node.y < -BASE_PADDING) {
    node.y = height + BASE_PADDING;
  } else if (node.y > height + BASE_PADDING) {
    node.y = -BASE_PADDING;
  }
}

function setCanvasSize(canvas, context, width, height) {
  const devicePixelRatio = Math.min(window.devicePixelRatio || 1, DEVICE_PIXEL_RATIO_CAP);
  canvas.width = Math.round(width * devicePixelRatio);
  canvas.height = Math.round(height * devicePixelRatio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

export default function NeuralNetworkBackground() {
  const canvasRef = React.useRef(null);
  const [reducedMotion, setReducedMotion] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const applyPreference = () => {
      setReducedMotion(mediaQuery.matches);
    };

    applyPreference();
    mediaQuery.addEventListener('change', applyPreference);

    return () => {
      mediaQuery.removeEventListener('change', applyPreference);
    };
  }, []);

  React.useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas || typeof window === 'undefined') {
      return undefined;
    }

    const context = canvas.getContext('2d');

    if (!context) {
      return undefined;
    }

    let viewportWidth = window.innerWidth;
    let viewportHeight = window.innerHeight;
    let renderMode = getRenderMode(viewportWidth);
    let nodes = [];
    let animationFrameId = 0;
    let previousTimestamp = 0;

    const rebuildScene = () => {
      viewportWidth = window.innerWidth;
      viewportHeight = window.innerHeight;
      renderMode = getRenderMode(viewportWidth);
      setCanvasSize(canvas, context, viewportWidth, viewportHeight);
      nodes = createNodes(viewportWidth, viewportHeight, renderMode);
    };

    const render = (deltaScale) => {
      context.clearRect(0, 0, viewportWidth, viewportHeight);

      for (const node of nodes) {
        node.x += node.vx * deltaScale;
        node.y += node.vy * deltaScale;
        wrapNode(node, viewportWidth, viewportHeight);
      }

      for (let index = 0; index < nodes.length; index += 1) {
        const sourceNode = nodes[index];

        for (let innerIndex = index + 1; innerIndex < nodes.length; innerIndex += 1) {
          const targetNode = nodes[innerIndex];
          const deltaX = sourceNode.x - targetNode.x;
          const deltaY = sourceNode.y - targetNode.y;
          const distance = Math.hypot(deltaX, deltaY);

          if (distance > renderMode.connectionDistance) {
            continue;
          }

          const opacity =
            Math.pow(1 - distance / renderMode.connectionDistance, 1.06) *
            renderMode.lineOpacity;
          const lineColor = blendColor(sourceNode.color, targetNode.color);

          context.beginPath();
          context.lineWidth = renderMode.lineWidth;
          context.strokeStyle = `rgba(${lineColor[0]}, ${lineColor[1]}, ${lineColor[2]}, ${opacity})`;
          context.moveTo(sourceNode.x, sourceNode.y);
          context.lineTo(targetNode.x, targetNode.y);
          context.stroke();
        }
      }

      for (const node of nodes) {
        const nodeAlpha = Math.min(1, node.alpha);
        const glowAlpha = Math.min(1, node.alpha + renderMode.glowBoost);
        const [red, green, blue] = node.color;

        context.beginPath();
        context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${glowAlpha})`;
        context.shadowBlur = renderMode.glowBlur;
        context.shadowColor = `rgba(${red}, ${green}, ${blue}, ${glowAlpha})`;
        context.arc(node.x, node.y, node.radius * 1.22, 0, Math.PI * 2);
        context.fill();

        context.beginPath();
        context.shadowBlur = 0;
        context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${nodeAlpha})`;
        context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        context.fill();
      }
    };

    const animate = (timestamp) => {
      if (previousTimestamp === 0) {
        previousTimestamp = timestamp;
      }

      const elapsed = timestamp - previousTimestamp;

      if (elapsed >= FRAME_INTERVAL) {
        previousTimestamp = timestamp - (elapsed % FRAME_INTERVAL);
        render(elapsed / FRAME_INTERVAL);
      }

      animationFrameId = window.requestAnimationFrame(animate);
    };

    rebuildScene();
    animationFrameId = window.requestAnimationFrame(animate);

    const handleResize = () => {
      rebuildScene();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className={styles.backgroundLayer} aria-hidden="true">
      <div className={styles.baseTone} />
      <canvas
        ref={canvasRef}
        className={`${styles.canvas} ${reducedMotion ? styles.canvasReducedMotion : ''}`}
      />
      <div className={styles.edgeTint} />
      <div className={styles.centerMask} />
    </div>
  );
}