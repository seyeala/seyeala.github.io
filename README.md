# TF.js Webcam Gesture Demo Template

This repo is a minimal starter for browser-based classification demos. Drop in a TensorFlow.js model exported from Colab, open the page (locally or via GitHub Pages), and watch live predictions from your webcam.

## Quick start
1. Copy your exported TF.js model files into `tfjs_model/` (or host them elsewhere) so the folder contains `model.json` and the shard `.bin` files.
2. Open `index.html` locally or at your GitHub Pages URL.
3. Enter the model URL (relative path works for GitHub Pages, e.g., `tfjs_model/model.json`) and your labels in output order (comma-separated).
4. Click **Load model & start camera**.

The page will request camera permission, load the model, and show live predictions at a modest frame rate to stay laptop-friendly.

## How it works
- `index.html` contains a single-page UI with inputs for the model URL and labels, a webcam preview, status text, and a live prediction display.
- `script.js` handles loading the TF.js model, starting the webcam, preprocessing frames to `IMAGE_SIZE` (default 160×160), running predictions every `PREDICTION_INTERVAL` milliseconds (default 200 ms), and mapping argmax outputs to the provided labels.
- Tensors are disposed after each prediction to avoid memory leaks, and you can adjust `IMAGE_SIZE` to match your training pipeline.

## Notes
- If you host models from another origin, ensure CORS headers allow browser access.
- To stop predictions, refresh or close the page; the loop only runs while the page is active.
- For GitHub Pages, ensure the repository is public and Pages is enabled so `index.html` is served.
