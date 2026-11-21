import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EMNIST_BALANCED_LABELS,
  computeCenterOfMassOffsets,
  findBoundingBox,
  formatPredictionResult,
  getCharacterCategory,
  rgbaToNormalizedPixels,
} from './preprocess.js';

test('EMNIST_BALANCED_LABELS contains exactly 47 unique characters', () => {
  assert.equal(EMNIST_BALANCED_LABELS.length, 47);
  assert.equal(new Set(EMNIST_BALANCED_LABELS).size, 47);
  assert.equal(EMNIST_BALANCED_LABELS[0], '0');
  assert.equal(EMNIST_BALANCED_LABELS[10], 'A');
  assert.equal(EMNIST_BALANCED_LABELS[41], 'g');
  assert.equal(EMNIST_BALANCED_LABELS[46], 't');
});

test('getCharacterCategory classifies digits, uppercase, and lowercase letters', () => {
  assert.equal(getCharacterCategory('7'), 'Digit');
  assert.equal(getCharacterCategory('A'), 'Uppercase Letter');
  assert.equal(getCharacterCategory('g'), 'Lowercase Letter');
});

test('findBoundingBox returns null on blank black canvas', () => {
  const rgba = new Uint8ClampedArray(280 * 280 * 4);
  for (let i = 3; i < rgba.length; i += 4) {
    rgba[i] = 255;
  }
  assert.equal(findBoundingBox(rgba, 280, 280), null);
});

test('findBoundingBox locates tight crop bounds of white stroke', () => {
  const width = 40;
  const height = 40;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 10; y <= 24; y++) {
    for (let x = 15; x <= 20; x++) {
      const idx = (y * width + x) * 4;
      rgba[idx] = 255;
      rgba[idx + 1] = 255;
      rgba[idx + 2] = 255;
      rgba[idx + 3] = 255;
    }
  }

  const box = findBoundingBox(rgba, width, height);
  assert.deepEqual(box, {
    minX: 15,
    minY: 10,
    maxX: 20,
    maxY: 24,
    cropWidth: 6,
    cropHeight: 15,
  });
});

test('computeCenterOfMassOffsets preserves 3px minimum margin for top-heavy character', () => {
  const w = 16;
  const h = 20;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      rgba[idx] = 255;
      rgba[idx + 1] = 255;
      rgba[idx + 2] = 255;
      rgba[idx + 3] = 255;
    }
  }
  const { offsetX, offsetY } = computeCenterOfMassOffsets(rgba, w, h);
  assert.ok(offsetX >= 3 && offsetX <= 28 - 3 - w);
  assert.ok(offsetY >= 3 && offsetY <= 28 - 3 - h);
});

test('rgbaToNormalizedPixels maps 0..255 grayscale to 0..1 float32 length 784', () => {
  const rgba = new Uint8ClampedArray(28 * 28 * 4);
  rgba[0] = 255;
  rgba[1] = 255;
  rgba[2] = 255;
  rgba[3] = 255;
  const pixels = rgbaToNormalizedPixels(rgba);
  assert.equal(pixels.length, 784);
  assert.equal(pixels[0], 1);
  assert.equal(pixels[1], 0);
});

test('formatPredictionResult selects highest EMNIST Balanced class and ranks candidates', () => {
  const probs = new Float32Array(47).fill(0.001);
  probs[10] = 0.8842; // 'A'
  probs[4] = 0.065;   // '4'
  const formatted = formatPredictionResult(probs);
  assert.equal(formatted.classIndex, 10);
  assert.equal(formatted.label, 'A');
  assert.equal(formatted.category, 'Uppercase Letter');
  assert.equal(formatted.confidencePercent, 88.42);
  assert.equal(formatted.ranked.length, 47);
  assert.equal(formatted.ranked[0].label, 'A');
  assert.equal(formatted.ranked[1].label, '4');
});
