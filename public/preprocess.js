export const TARGET_IMAGE_SIZE = 28;
export const BOUNDING_BOX_FIT_SIZE = 20;
export const FOREGROUND_THRESHOLD = 20;

/**
 * Official 47 classes in NIST EMNIST Balanced (emnist-balanced-mapping.txt):
 * 0-9 (digits), A-Z (uppercase), and 11 distinct lowercase letters.
 */
export const EMNIST_BALANCED_LABELS = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z',
  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't',
];

/**
 * Returns human-readable character category for EMNIST labels.
 * @param {string} label
 * @returns {'Digit' | 'Uppercase Letter' | 'Lowercase Letter'}
 */
export function getCharacterCategory(label) {
  if (/^[0-9]$/.test(label)) return 'Digit';
  if (/^[A-Z]$/.test(label)) return 'Uppercase Letter';
  return 'Lowercase Letter';
}

/**
 * Finds the tight bounding box of bright foreground pixels in an RGBA buffer.
 * Returns null if the canvas contains no strokes above threshold.
 *
 * @param {Uint8ClampedArray | Uint8Array} rgbaData
 * @param {number} width
 * @param {number} height
 * @returns {{ minX: number, minY: number, maxX: number, maxY: number, cropWidth: number, cropHeight: number } | null}
 */
export function findBoundingBox(rgbaData, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const intensity = (rgbaData[idx] + rgbaData[idx + 1] + rgbaData[idx + 2]) / 3;
      if (intensity > FOREGROUND_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return null;
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    cropWidth: maxX - minX + 1,
    cropHeight: maxY - minY + 1,
  };
}

/**
 * Computes proportional 20x20 target dimensions and MNIST/EMNIST-aligned
 * center-of-mass placement offsets inside a 28x28 frame with a 3px minimum margin.
 *
 * @param {Uint8ClampedArray | Uint8Array} scaledRgba
 * @param {number} scaledWidth
 * @param {number} scaledHeight
 * @returns {{ offsetX: number, offsetY: number }}
 */
export function computeCenterOfMassOffsets(scaledRgba, scaledWidth, scaledHeight) {
  let massSum = 0;
  let weightedX = 0;
  let weightedY = 0;

  for (let y = 0; y < scaledHeight; y++) {
    for (let x = 0; x < scaledWidth; x++) {
      const idx = (y * scaledWidth + x) * 4;
      const weight = (scaledRgba[idx] + scaledRgba[idx + 1] + scaledRgba[idx + 2]) / 3;
      massSum += weight;
      weightedX += x * weight;
      weightedY += y * weight;
    }
  }

  const comX = massSum > 0 ? weightedX / massSum : scaledWidth / 2;
  const comY = massSum > 0 ? weightedY / massSum : scaledHeight / 2;

  const targetCenter = TARGET_IMAGE_SIZE / 2;
  const rawOffsetX = Math.round(targetCenter - comX);
  const rawOffsetY = Math.round(targetCenter - comY);

  const minMarginX = scaledWidth <= TARGET_IMAGE_SIZE - 6 ? 3 : 0;
  const minMarginY = scaledHeight <= TARGET_IMAGE_SIZE - 6 ? 3 : 0;

  return {
    offsetX: Math.max(minMarginX, Math.min(TARGET_IMAGE_SIZE - minMarginX - scaledWidth, rawOffsetX)),
    offsetY: Math.max(minMarginY, Math.min(TARGET_IMAGE_SIZE - minMarginY - scaledHeight, rawOffsetY)),
  };
}

/**
 * Converts a 28x28 RGBA buffer into a normalized [0, 1] Float32Array of length 784.
 *
 * @param {Uint8ClampedArray | Uint8Array} rgba28
 * @returns {Float32Array}
 */
export function rgbaToNormalizedPixels(rgba28) {
  const expectedLength = TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE * 4;
  if (!rgba28 || rgba28.length !== expectedLength) {
    throw new Error(`Expected ${expectedLength} RGBA bytes for 28x28 image.`);
  }

  const pixels = new Float32Array(TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE);
  for (let i = 0; i < TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE; i++) {
    const idx = i * 4;
    const gray = (rgba28[idx] + rgba28[idx + 1] + rgba28[idx + 2]) / 3;
    pixels[i] = gray / 255.0;
  }
  return pixels;
}

/**
 * Formats a 47-class EMNIST Balanced softmax probability vector into a structured
 * prediction result with ranked candidates.
 *
 * @param {ArrayLike<number>} probabilities
 * @param {string[]} [labels]
 */
export function formatPredictionResult(probabilities, labels = EMNIST_BALANCED_LABELS) {
  if (!probabilities || probabilities.length !== labels.length) {
    throw new Error(`Prediction probabilities must contain exactly ${labels.length} class values.`);
  }

  const ranked = new Array(labels.length);
  for (let i = 0; i < labels.length; i++) {
    const val = Number(probabilities[i]);
    if (!Number.isFinite(val)) {
      throw new Error(`Invalid probability value at class index ${i}.`);
    }
    const label = labels[i];
    ranked[i] = {
      index: i,
      label,
      category: getCharacterCategory(label),
      probability: val,
    };
  }

  ranked.sort((a, b) => b.probability - a.probability);
  const top = ranked[0];
  const confidence = Math.max(0, Math.min(1, top.probability));
  const confidencePercent = Number((confidence * 100).toFixed(2));

  return {
    classIndex: top.index,
    label: top.label,
    category: top.category,
    confidence,
    confidencePercent,
    ranked,
  };
}
