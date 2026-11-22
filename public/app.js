import {
  BOUNDING_BOX_FIT_SIZE,
  EMNIST_BALANCED_LABELS,
  TARGET_IMAGE_SIZE,
  computeCenterOfMassOffsets,
  findBoundingBox,
  formatPredictionResult,
  getCharacterCategory,
  rgbaToNormalizedPixels,
} from './preprocess.js';

const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('preview-canvas');
const previewCtx = previewCanvas.getContext('2d', { willReadFrequently: true });

const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const sampleBtn = document.getElementById('sample-btn');
const feedbackEl = document.getElementById('canvas-feedback');

const predictedDigitEl = document.getElementById('predicted-digit');
const predictedCategoryEl = document.getElementById('predicted-category');
const confidenceValueEl = document.getElementById('confidence-value');
const confidenceBarEl = document.getElementById('confidence-bar');
const confidenceTrackEl = document.getElementById('confidence-track');
const latencyReadoutEl = document.getElementById('latency-readout');
const inferenceMetaEl = document.getElementById('inference-meta');
const probListEl = document.getElementById('prob-list');
const distHeadingEl = document.getElementById('dist-heading');
const viewTop10Btn = document.getElementById('view-top10-btn');
const viewAll47Btn = document.getElementById('view-all47-btn');

const architectureBodyEl = document.getElementById('architecture-body');
const paramCountMetaEl = document.getElementById('param-count-meta');
const historyBodyEl = document.getElementById('history-body');
const evalSummaryMetaEl = document.getElementById('eval-summary-meta');
const kpiTestAccEl = document.getElementById('kpi-test-acc');
const kpiTestLossEl = document.getElementById('kpi-test-loss');
const kpiClassesEl = document.getElementById('kpi-classes');
const kpiParamsEl = document.getElementById('kpi-params');

let cnnModel = null;
let classLabels = EMNIST_BALANCED_LABELS;
let isDrawing = false;
let hasStrokes = false;
let lastPoint = null;
let sampleIndex = 0;
let showAllClasses = false;
let lastPrediction = null;

/**
 * Initializes the 280x280 drawing canvas with black background and white thick brush.
 */
function initCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);

  hasStrokes = false;
  lastPoint = null;
}

/**
 * Renders either Top 10 ranked candidates or all 47 EMNIST Balanced class bars.
 */
function renderProbabilityList(prediction) {
  probListEl.innerHTML = '';
  probListEl.classList.toggle('is-expanded', showAllClasses);
  distHeadingEl.textContent = showAllClasses
    ? `All ${classLabels.length} EMNIST Balanced Classes`
    : 'Top 10 Predicted Candidates';

  let items;
  if (prediction) {
    items = showAllClasses
      ? classLabels.map((label, idx) => {
        const match = prediction.ranked.find((r) => r.index === idx);
        return match ?? { index: idx, label, category: getCharacterCategory(label), probability: 0 };
      })
      : prediction.ranked.slice(0, 10);
  } else {
    const idleSlice = showAllClasses ? classLabels : classLabels.slice(0, 10);
    items = idleSlice.map((label, idx) => ({
      index: idx,
      label,
      category: getCharacterCategory(label),
      probability: 0,
    }));
  }

  for (const item of items) {
    const row = document.createElement('div');
    const isWinner = Boolean(prediction && item.index === prediction.classIndex);
    row.className = isWinner ? 'prob-row is-winner' : 'prob-row';
    row.title = `${item.label} (${item.category})`;

    const labelEl = document.createElement('span');
    labelEl.className = 'prob-digit';
    labelEl.textContent = item.label;

    const track = document.createElement('div');
    track.className = 'prob-bar-track';

    const fill = document.createElement('div');
    fill.className = 'prob-bar-fill';
    fill.style.transform = `scaleX(${Math.max(0, Math.min(1, item.probability))})`;
    track.appendChild(fill);

    const pct = document.createElement('span');
    pct.className = 'prob-pct';
    pct.textContent = `${(item.probability * 100).toFixed(2)}%`;

    row.append(labelEl, track, pct);
    probListEl.appendChild(row);
  }
}

/**
 * Resets the prediction output UI back to its idle ready state.
 */
function resetResults() {
  lastPrediction = null;
  predictedDigitEl.textContent = '—';
  predictedDigitEl.classList.remove('has-prediction');
  predictedCategoryEl.textContent = 'Awaiting input';
  confidenceValueEl.textContent = '0.00%';
  confidenceBarEl.style.transform = 'scaleX(0)';
  confidenceTrackEl.setAttribute('aria-valuenow', '0');
  latencyReadoutEl.textContent = cnnModel
    ? `Ready (${window.tf.getBackend()} backend)`
    : 'Initializing browser runtime';
  inferenceMetaEl.textContent = `Shape: [1, 28, 28, 1] · ${classLabels.length} Classes`;

  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);

  renderProbabilityList(null);
}

function setFeedback(message, isError = false) {
  feedbackEl.textContent = message;
  feedbackEl.classList.toggle('is-error', isError);
}

/**
 * Translates pointer coordinates into exact canvas pixel coordinates.
 */
function getCanvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function handlePointerDown(event) {
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  isDrawing = true;
  hasStrokes = true;
  const point = getCanvasCoords(event);
  lastPoint = point;

  ctx.beginPath();
  ctx.arc(point.x, point.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  setFeedback('Stroke captured. Click Predict when finished.');
}

function handlePointerMove(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const point = getCanvasCoords(event);

  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();

  lastPoint = point;
}

function handlePointerUp(event) {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

/**
 * Preprocesses the 280x280 drawing canvas directly in the browser:
 * crops to tight bounding box, scales proportionally into 20x20, centers by
 * center-of-mass inside a 28x28 offscreen canvas, and normalizes to Float32Array(784).
 */
function preprocessCanvasInBrowser() {
  const sourceImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const box = findBoundingBox(sourceImageData.data, canvas.width, canvas.height);

  if (!box) {
    return {
      isBlank: true,
      pixels: new Float32Array(TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE),
    };
  }

  const scale = BOUNDING_BOX_FIT_SIZE / Math.max(box.cropWidth, box.cropHeight);
  const scaledWidth = Math.max(1, Math.round(box.cropWidth * scale));
  const scaledHeight = Math.max(1, Math.round(box.cropHeight * scale));

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = scaledWidth;
  scaledCanvas.height = scaledHeight;
  const scaledCtx = scaledCanvas.getContext('2d', { willReadFrequently: true });
  scaledCtx.fillStyle = '#000000';
  scaledCtx.fillRect(0, 0, scaledWidth, scaledHeight);
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(
    canvas,
    box.minX,
    box.minY,
    box.cropWidth,
    box.cropHeight,
    0,
    0,
    scaledWidth,
    scaledHeight
  );

  const scaledImageData = scaledCtx.getImageData(0, 0, scaledWidth, scaledHeight);
  const { offsetX, offsetY } = computeCenterOfMassOffsets(
    scaledImageData.data,
    scaledWidth,
    scaledHeight
  );

  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
  previewCtx.drawImage(scaledCanvas, offsetX, offsetY);

  const final28 = previewCtx.getImageData(0, 0, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);
  const pixels = rgbaToNormalizedPixels(final28.data);

  return { isBlank: false, pixels };
}

/**
 * Updates the Results panel with predicted character, category, confidence bar, and ranked probabilities.
 */
function renderPrediction(result, inferenceTimeMs) {
  lastPrediction = result;
  predictedDigitEl.textContent = result.label;
  predictedDigitEl.classList.add('has-prediction');
  predictedCategoryEl.textContent = result.category;

  const pctText = `${result.confidencePercent.toFixed(2)}%`;
  confidenceValueEl.textContent = pctText;
  confidenceBarEl.style.transform = `scaleX(${Math.max(0, Math.min(1, result.confidence))})`;
  confidenceTrackEl.setAttribute('aria-valuenow', String(Math.round(result.confidencePercent)));

  latencyReadoutEl.textContent = `Browser inference: ${inferenceTimeMs.toFixed(2)} ms (${window.tf.getBackend()})`;
  inferenceMetaEl.textContent = `Shape: [1, 28, 28, 1] · ${classLabels.length} Classes`;

  renderProbabilityList(result);
}

/**
 * Runs 47-class CNN inference entirely in the browser using TensorFlow.js.
 */
function predictCharacter() {
  if (!cnnModel) {
    setFeedback('CNN model is still loading in your browser. Please wait a moment.', true);
    return;
  }

  if (!hasStrokes) {
    setFeedback('Canvas is empty. Draw a digit or letter before predicting.', true);
    return;
  }

  const { isBlank, pixels } = preprocessCanvasInBrowser();
  if (isBlank) {
    setFeedback('The drawing canvas is blank. Draw a digit or letter before predicting.', true);
    return;
  }

  const startTime = performance.now();

  const rawProbabilities = window.tf.tidy(() => {
    const inputTensor = window.tf.tensor4d(pixels, [1, 28, 28, 1], 'float32');
    const outputTensor = cnnModel.predict(inputTensor);
    return Float32Array.from(outputTensor.dataSync());
  });

  const inferenceTimeMs = Number((performance.now() - startTime).toFixed(2));
  const result = formatPredictionResult(rawProbabilities, classLabels);

  renderPrediction(result, inferenceTimeMs);
  setFeedback(
    `Predicted "${result.label}" (${result.category}) with ${result.confidencePercent.toFixed(2)}% confidence.`
  );
}

/**
 * Draws crisp synthetic handwritten characters (letters and digits) onto the canvas.
 */
function drawSampleCharacter() {
  initCanvas();
  const samples = [
    // Uppercase 'A'
    () => {
      ctx.beginPath();
      ctx.moveTo(76, 224);
      ctx.lineTo(140, 56);
      ctx.lineTo(204, 224);
      ctx.moveTo(98, 160);
      ctx.lineTo(182, 160);
      ctx.stroke();
    },
    // Digit '7'
    () => {
      ctx.beginPath();
      ctx.moveTo(78, 66);
      ctx.lineTo(202, 66);
      ctx.lineTo(120, 224);
      ctx.stroke();
    },
    // Uppercase 'E'
    () => {
      ctx.beginPath();
      ctx.moveTo(190, 66);
      ctx.lineTo(94, 66);
      ctx.lineTo(94, 214);
      ctx.lineTo(194, 214);
      ctx.moveTo(94, 140);
      ctx.lineTo(174, 140);
      ctx.stroke();
    },
    // Uppercase 'B'
    () => {
      ctx.beginPath();
      ctx.moveTo(94, 64);
      ctx.lineTo(94, 216);
      ctx.moveTo(94, 64);
      ctx.bezierCurveTo(192, 64, 192, 136, 94, 136);
      ctx.bezierCurveTo(202, 136, 202, 216, 94, 216);
      ctx.stroke();
    },
    // Digit '3'
    () => {
      ctx.beginPath();
      ctx.moveTo(92, 74);
      ctx.bezierCurveTo(188, 52, 198, 128, 130, 138);
      ctx.bezierCurveTo(206, 144, 198, 226, 88, 214);
      ctx.stroke();
    },
  ];

  samples[sampleIndex % samples.length]();
  sampleIndex += 1;
  hasStrokes = true;
}

function formatLayerConfig(layer) {
  const config = layer.getConfig ? layer.getConfig() : {};
  const parts = [];
  if (config.filters) parts.push(`filters=${config.filters}`);
  if (config.kernelSize) {
    parts.push(
      `kernel=${Array.isArray(config.kernelSize) ? config.kernelSize.join('×') : config.kernelSize}`
    );
  }
  if (config.poolSize) {
    parts.push(
      `pool=${Array.isArray(config.poolSize) ? config.poolSize.join('×') : config.poolSize}`
    );
  }
  if (config.units) parts.push(`units=${config.units}`);
  if (config.rate !== null && config.rate !== undefined) parts.push(`rate=${config.rate}`);
  if (config.activation) parts.push(`act=${config.activation}`);
  return parts.length > 0 ? parts.join(', ') : '—';
}

function renderArchitectureFromModel(model) {
  const totalParams = model.countParams();
  paramCountMetaEl.textContent = `${totalParams.toLocaleString()} trainable parameters`;
  kpiParamsEl.textContent = totalParams.toLocaleString();

  architectureBodyEl.innerHTML = model.layers
    .map((layer, idx) => {
      const shape = Array.isArray(layer.outputShape)
        ? `[${layer.outputShape.map((d) => (d === null ? 'batch' : d)).join(', ')}]`
        : String(layer.outputShape);

      return `
        <tr>
          <td class="mono-cell">${idx + 1}</td>
          <td class="mono-cell">${layer.name}</td>
          <td>${layer.getClassName()}</td>
          <td class="mono-cell">${formatLayerConfig(layer)}</td>
          <td class="mono-cell">${shape}</td>
          <td class="num-col">${layer.countParams().toLocaleString()}</td>
        </tr>
      `;
    })
    .join('');
}

async function loadStaticEvaluationMetrics() {
  try {
    let response = await fetch('./emnist_model/training_metrics.json');
    let loadedEmnistMetrics = response.ok;
    if (!response.ok) {
      response = await fetch('./tfjs_model/training_metrics.json');
    }
    if (!response.ok) return;
    const m = await response.json();

    if (loadedEmnistMetrics && Array.isArray(m.classLabels) && m.classLabels.length === 47) {
      classLabels = m.classLabels;
    } else {
      classLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    }

    kpiTestAccEl.textContent = `${Number(m.testAccuracy).toFixed(2)}%`;
    kpiTestLossEl.textContent = Number(m.testLoss).toFixed(4);
    kpiClassesEl.textContent = String(m.numClasses ?? classLabels.length);
    evalSummaryMetaEl.textContent = `${m.optimizer} · ${m.trainSamples.toLocaleString()} train / ${m.testSamples.toLocaleString()} test`;

    historyBodyEl.innerHTML = m.history
      .map(
        (row) => `
        <tr>
          <td class="mono-cell">Epoch ${row.epoch}</td>
          <td class="num-col">${Number(row.loss).toFixed(4)}</td>
          <td class="num-col">${Number(row.accuracy).toFixed(2)}%</td>
          <td class="num-col">${Number(row.valLoss).toFixed(4)}</td>
          <td class="num-col">${Number(row.valAccuracy).toFixed(2)}%</td>
        </tr>
      `
      )
      .join('');
  } catch (err) {
    console.warn('Could not load static training metrics:', err);
  }
}

function ensureTensorFlowReady() {
  if (window.tf) return Promise.resolve(window.tf);

  const sources = [
    './vendor/tf.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
  ];

  const loadScript = (src) => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => window.tf
      ? resolve(window.tf)
      : reject(new Error(`TensorFlow.js loaded without window.tf: ${src}`));
    script.onerror = () => reject(new Error(`Failed to load TensorFlow.js: ${src}`));
    document.head.appendChild(script);
  });

  return sources.reduce(
    (promise, src) => promise.catch(() => loadScript(src)),
    Promise.reject(new Error('Start TensorFlow.js loading'))
  ).then((tf) => tf.ready()).then(() => window.tf);
}

async function initBrowserModel() {
  try {
    const tf = await ensureTensorFlowReady();
    await tf.ready();
    let emnistModel = true;
    try {
      cnnModel = await tf.loadLayersModel('./emnist_model/model.json');
    } catch (emnistError) {
      console.warn('EMNIST model unavailable; loading the available digit model.', emnistError);
      emnistModel = false;
      cnnModel = await tf.loadLayersModel('./tfjs_model/model.json');
      classLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    }

    const outputShape = cnnModel.outputs?.[0]?.shape;
    const outputClasses = outputShape?.[outputShape.length - 1];
    if (outputClasses !== classLabels.length) {
      throw new Error(
        `Invalid model output: found ${outputClasses} classes, expected ${classLabels.length}.`
      );
    }

    tf.tidy(() => {
      const warmup = tf.zeros([1, 28, 28, 1]);
      cnnModel.predict(warmup);
    });

    renderArchitectureFromModel(cnnModel);
    predictBtn.disabled = false;
    latencyReadoutEl.textContent = `Ready (${tf.getBackend()} backend)`;
    setFeedback(
      emnistModel
        ? 'Draw a digit or letter (0–9, A–Z, a–t) centered in the black canvas.'
        : 'Digit model ready (0–9). Letters require the 47-class EMNIST model.'
    );
  } catch (err) {
    console.error('Failed to load TensorFlow.js model:', err);
    setFeedback('Failed to load EMNIST Balanced model files from ./emnist_model/model.json.', true);
  }
}

// Event listeners
canvas.addEventListener('pointerdown', handlePointerDown);
canvas.addEventListener('pointermove', handlePointerMove);
canvas.addEventListener('pointerup', handlePointerUp);
canvas.addEventListener('pointercancel', handlePointerUp);

predictBtn.addEventListener('click', predictCharacter);
clearBtn.addEventListener('click', () => {
  initCanvas();
  resetResults();
  setFeedback('Canvas cleared. Draw a digit or letter centered in the black frame.');
});
sampleBtn.addEventListener('click', () => {
  drawSampleCharacter();
  predictCharacter();
});

viewTop10Btn.addEventListener('click', () => {
  showAllClasses = false;
  viewTop10Btn.classList.add('is-active');
  viewTop10Btn.setAttribute('aria-pressed', 'true');
  viewAll47Btn.classList.remove('is-active');
  viewAll47Btn.setAttribute('aria-pressed', 'false');
  renderProbabilityList(lastPrediction);
});

viewAll47Btn.addEventListener('click', () => {
  showAllClasses = true;
  viewAll47Btn.classList.add('is-active');
  viewAll47Btn.setAttribute('aria-pressed', 'true');
  viewTop10Btn.classList.remove('is-active');
  viewTop10Btn.setAttribute('aria-pressed', 'false');
  renderProbabilityList(lastPrediction);
});

// Initialize on load
initCanvas();
resetResults();
loadStaticEvaluationMetrics().then(() => {
  if (!lastPrediction) {
    renderProbabilityList(null);
  }
});
initBrowserModel();
