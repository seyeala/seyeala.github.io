const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediction');
const videoEl = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');
const modelUrlInput = document.getElementById('model-url');

const IMAGE_SIZE = 160; // must match your model's input size

const DEFAULT_MODEL_URL = 'tfjs_model/model.json';

let model = null;
let running = false;
let hookOverrides = null;

function getActiveHooks() {
  if (hookOverrides) return hookOverrides;

  return {
    loadModel,
    setupCamera,
    predictLoop,
  };
}

function validateManifest(manifest, url) {
  if (!manifest || !Array.isArray(manifest.weightsManifest) || manifest.weightsManifest.length === 0) {
    return `Model file at "${url}" is missing a weightsManifest array. Check that you exported the full model.json + shard files.`;
  }

  const invalidGroup = manifest.weightsManifest.find(
    group => !Array.isArray(group.paths) || group.paths.length === 0 || !group.paths.every(p => typeof p === 'string' && p.trim())
  );

  if (invalidGroup) {
    return (
      `Model file at "${url}" has an invalid weightsManifest.paths entry. ` +
      'Each weightsManifest item must include a non-empty array of shard file names. Re-export the model to regenerate model.json.'
    );
  }

  return null;
}

function getModelUrlFromInput() {
  const provided = modelUrlInput?.value?.trim();
  return provided || DEFAULT_MODEL_URL;
}

async function loadModel() {
  const modelUrl = getModelUrlFromInput();
  const resolvedModelPath = new URL(modelUrl, window.location.href).toString();
  let manifest;
  let resp;

  statusEl.textContent = `Loading model from ${modelUrl}...`;

  try {
    resp = await fetch(resolvedModelPath);
  } catch (err) {
    throw new Error(`${modelUrl} -> network error (${err.message})`);
  }

  if (!resp.ok) {
    throw new Error(`${modelUrl} -> HTTP ${resp.status}`);
  }

  try {
    manifest = await resp.json();
  } catch (err) {
    throw new Error(`${modelUrl} -> invalid JSON`);
  }

  const validationError = validateManifest(manifest, modelUrl);
  if (validationError) {
    throw new Error(`${modelUrl} -> ${validationError}`);
  }

  statusEl.textContent = `Loading model from ${resolvedModelPath}...`;
  model = await tf.loadLayersModel(resolvedModelPath);

  statusEl.textContent = 'Model loaded. Requesting camera...';
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoEl.srcObject = stream;

  await new Promise(resolve => {
    videoEl.onloadedmetadata = () => {
      videoEl.play();
      resolve();
    };
  });

  statusEl.textContent = 'Camera ready. Running predictions...';
}

function predictLoop() {
  if (!running || !model) return;

  tf.tidy(() => {
    // Grab frame from video
    const img = tf.browser.fromPixels(videoEl)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE]) // [H,W,3]
      .toFloat()
      .expandDims(0); // [1,H,W,3]

    // IMPORTANT: your model already normalizes inside (x/127.5 - 1),
    // so we do NOT divide by 255 here.

    const logits = model.predict(img);
    const probs = logits.dataSync();
    const maxIdx = probs.indexOf(Math.max(...probs));

    // For now, just show the index; you can map to labels later.
    predEl.textContent = `class_${maxIdx}`;
  });

  requestAnimationFrame(predictLoop);
}

async function start() {
  if (running) return;
  running = true;

  const hooks = getActiveHooks();

  try {
    await hooks.loadModel();
    await hooks.setupCamera();
    hooks.predictLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
    running = false;
  }
}

startBtn.addEventListener('click', () => start());

function setTestHooks(overrides) {
  hookOverrides = {
    loadModel,
    setupCamera,
    predictLoop,
    ...overrides,
  };
}

function resetTestHooks() {
  hookOverrides = null;
  running = false;
  model = null;
  statusEl.textContent = 'Idle';
  predEl.textContent = '-';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DEFAULT_MODEL_URL,
    getModelUrlFromInput,
    validateManifest,
    start,
    setTestHooks,
    resetTestHooks,
  };
}
