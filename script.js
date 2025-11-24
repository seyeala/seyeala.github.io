const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediction');
const videoEl = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');

const IMAGE_SIZE = 160; // must match your model's input size

// Try a few common locations for the TF.js export. This helps when the page is
// hosted from a subfolder (e.g., GitHub Pages) but the model files live one
// directory up from the HTML page.
const MODEL_CANDIDATES = [
  './tfjs_model/model.json',
  '../tfjs_model/model.json',
  '/tfjs_model/model.json',
];

let model = null;
let running = false;

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

async function loadModel() {
  const errors = [];
  let resolvedModelPath = null;

  // Fetch the JSON up front so we can validate its contents and surface
  // clearer errors than the generic "Array.prototype.every called on null".
  for (const candidate of MODEL_CANDIDATES) {
    const url = new URL(candidate, window.location.href).toString();
    statusEl.textContent = `Looking for model at ${url}...`;

    let manifest;
    let resp;
    try {
      resp = await fetch(url);
    } catch (err) {
      errors.push(`${url} -> network error (${err.message})`);
      continue;
    }

    if (!resp.ok) {
      errors.push(`${url} -> HTTP ${resp.status}`);
      continue;
    }

    try {
      manifest = await resp.json();
    } catch (err) {
      errors.push(`${url} -> invalid JSON`);
      continue;
    }

    const validationError = validateManifest(manifest, url);
    if (validationError) {
      errors.push(`${url} -> ${validationError}`);
      continue;
    }

    resolvedModelPath = url;
    break;
  }

  if (!resolvedModelPath) {
    throw new Error(
      'Model file not found. Tried the following locations:\n' + errors.join('\n') +
      '\nPlace your exported TF.js files inside ./tfjs_model/ (next to index.html) or adjust MODEL_CANDIDATES accordingly.'
    );
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

  try {
    await loadModel();
    await setupCamera();
    predictLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
    running = false;
  }
}

startBtn.addEventListener('click', start);
