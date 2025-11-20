const statusEl = document.getElementById('status');
const predEl = document.getElementById('prediction');
const videoEl = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');

const IMAGE_SIZE = 160; // must match your model's input size
const MODEL_PATH = 'tfjs_model/model.json';

let model = null;
let running = false;

async function loadModel() {
  statusEl.textContent = 'Loading model...';

  // Fetch the JSON up front so we can validate its contents and surface
  // clearer errors than the generic "Array.prototype.every called on null".
  const resp = await fetch(MODEL_PATH);
  if (!resp.ok) {
    throw new Error(
      `Model file not found at "${MODEL_PATH}" (HTTP ${resp.status}). ` +
      'Place your exported TF.js files inside ./tfjs_model/ and ensure the main file is named model.json.'
    );
  }

  let manifest;
  try {
    manifest = await resp.json();
  } catch (err) {
    throw new Error(
      `Model file at "${MODEL_PATH}" is not valid JSON. Re-export the model or fix the file contents.`
    );
  }

  if (!manifest || !Array.isArray(manifest.weightsManifest) || manifest.weightsManifest.length === 0) {
    throw new Error(
      `Model file at "${MODEL_PATH}" is missing a weightsManifest array. Check that you exported the full model.json + shard files.`
    );
  }

  model = await tf.loadLayersModel(MODEL_PATH);

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
