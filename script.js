let model = null;
let labels = [];
let isRunning = false;
let isGraphModel = false;

const IMAGE_SIZE = 160; // must match what you used in Colab
const PREDICTION_INTERVAL = 200; // ms between predictions

const statusEl = document.getElementById('status-text');
const predictionEl = document.getElementById('prediction');
const logEl = document.getElementById('log');
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('offscreen-canvas');
const ctx = canvasEl.getContext('2d');

function log(msg) {
  console.log(msg);
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

async function loadModel() {
  const modelUrl = document.getElementById('model-url').value.trim();
  const labelsText = document.getElementById('labels').value.trim();

  if (!modelUrl) {
    alert('Please specify a model URL.');
    return null;
  }

  labels = labelsText ? labelsText.split(',').map(s => s.trim()).filter(Boolean) : [];

  statusEl.textContent = 'Loading model...';
  log(`Loading model from: ${modelUrl}`);

  let modelMetadata;
  try {
    const response = await fetch(modelUrl, { cache: 'no-cache' });

    if (!response.ok) {
      throw new Error(`Model JSON could not be fetched (HTTP ${response.status}).`);
    }

    modelMetadata = await response.json();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusEl.textContent = 'Failed to load model.';
    log(`Error retrieving model file: ${message}`);
    return null;
  }

  const format = modelMetadata?.format?.toLowerCase?.();
  const looksGraphModel = format?.includes('graph') || Boolean(modelMetadata?.graphdefSignature);
  const looksLayersModel = format?.includes('layers') || Boolean(modelMetadata?.modelTopology);

  try {
    const preferGraph = looksGraphModel && !looksLayersModel;

    const loadLayers = async () => {
      const loadedModel = await tf.loadLayersModel(modelUrl);
      model = loadedModel;
      isGraphModel = false;
      statusEl.textContent = 'Model loaded.';
      log('Model loaded successfully as a Layers model.');
      return model;
    };

    const loadGraph = async () => {
      const loadedGraphModel = await tf.loadGraphModel(modelUrl);
      model = loadedGraphModel;
      isGraphModel = true;
      statusEl.textContent = 'Model loaded.';
      log('Model loaded successfully as a GraphModel.');
      return model;
    };

    if (preferGraph) {
      try {
        return await loadGraph();
      } catch (graphErr) {
        log('GraphModel load failed, falling back to Layers loader...');
        console.error(graphErr);
        return await loadLayers();
      }
    }

    try {
      return await loadLayers();
    } catch (layersErr) {
      log('Layers model load failed, trying GraphModel...');
      console.error(layersErr);
      return await loadGraph();
    }
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Failed to load model.';
    const message = err instanceof Error ? err.message : String(err);
    log('Error loading model: ' + message);
    if (message.includes('producer')) {
      log('Tip: This often happens when the URL does not point to a valid TF.js model.json file.');
    }
    return null;
  }
}

async function setupCamera() {
  statusEl.textContent = 'Requesting camera access...';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl.srcObject = stream;
    await new Promise(resolve => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });
    statusEl.textContent = 'Camera ready.';
    log('Camera started.');
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Camera access denied or not available.';
    log('Error starting camera: ' + err.message);
    throw err;
  }
}

function preprocessFrame() {
  // Draw current video frame to canvas
  ctx.drawImage(videoEl, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  // Convert to tensor and preprocess
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(canvasEl); // [H, W, 3]
    img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
    img = img.toFloat().div(255.0); // [0,1]
    img = img.expandDims(0); // [1, H, W, 3]
    return img;
  });
}

async function predictLoop() {
  if (!model || !isRunning) return;

  const input = preprocessFrame();

  const logits = tf.tidy(() => {
    if (isGraphModel) {
      const inputName = model.inputs?.[0]?.name;
      const outputName = model.outputs?.[0]?.name;

      if (!inputName) {
        throw new Error('Graph model is missing input name; ensure a proper signature was exported.');
      }

      const result = model.execute({ [inputName]: input }, outputName);
      return Array.isArray(result) ? result[0] : result;
    }

    return model.predict(input);
  });
  const probs = await logits.data();
  const maxIdx = probs.indexOf(Math.max(...probs));
  const label = labels[maxIdx] || `class_${maxIdx}`;

  predictionEl.textContent = label;

  tf.dispose([input, logits]);

  // Schedule next prediction
  setTimeout(predictLoop, PREDICTION_INTERVAL);
}

async function start() {
  if (isRunning) return;
  isRunning = true;

  // 1) Load model
  const loaded = await loadModel();
  if (!loaded) {
    isRunning = false;
    return;
  }

  // 2) Setup camera
  try {
    await setupCamera();
  } catch {
    isRunning = false;
    return;
  }

  // 3) Start prediction loop
  statusEl.textContent = 'Running predictions...';
  predictLoop();
}

document.getElementById('start-btn').addEventListener('click', () => {
  start();
});
