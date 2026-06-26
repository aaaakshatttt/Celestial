# app.py
# PURPOSE: A simple web app where you upload an IR image
#          and get the colorized RGB result back instantly.
#
# Run with: python app.py
# Open:     http://localhost:5000

import os
import io
import time
import numpy as np
import torch
import cv2
from flask import Flask, request, jsonify, send_file, render_template_string
from src.model import Generator

app = Flask(__name__)

CHECKPOINT_PATH = "outputs/checkpoints"
IMAGE_SIZE      = 256
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────
def load_model():
    from glob import glob
    files = sorted(glob(os.path.join(CHECKPOINT_PATH, "*.pth")))
    if not files:
        print("WARNING: No checkpoint found. Train the model first.")
        return None
    G = Generator().to(device)
    checkpoint = torch.load(files[-1], map_location=device)
    G.load_state_dict(checkpoint["G_state"])
    G.eval()
    print(f"Model loaded from: {files[-1]}")
    return G

G = load_model()


# ─────────────────────────────────────────────
# HTML PAGE
# ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Celestial — IR Colorization</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f0f11;
      color: #e8e8e8;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }

    .container {
      width: 100%;
      max-width: 900px;
    }

    .header {
      text-align: center;
      margin-bottom: 2.5rem;
    }

    .header h1 {
      font-size: 2rem;
      font-weight: 600;
      background: linear-gradient(135deg, #6ee7f7, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }

    .header p {
      color: #888;
      font-size: 0.95rem;
    }

    .card {
      background: #1a1a1f;
      border: 1px solid #2a2a35;
      border-radius: 16px;
      padding: 1.5rem;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    .panel label {
      font-size: 0.75rem;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: block;
      margin-bottom: 0.75rem;
    }

    .drop-zone {
      border: 2px dashed #2a2a35;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      background: #12121a;
    }

    .drop-zone:hover, .drop-zone.drag-over {
      border-color: #6ee7f7;
      background: #1a2a30;
    }

    .drop-zone .icon {
      font-size: 2rem;
      margin-bottom: 0.75rem;
      display: block;
    }

    .drop-zone p {
      color: #888;
      font-size: 0.85rem;
    }

    .drop-zone .formats {
      font-size: 0.75rem;
      color: #555;
      margin-top: 0.25rem;
    }

    .image-preview {
      border-radius: 12px;
      overflow: hidden;
      background: #12121a;
      border: 1px solid #2a2a35;
      aspect-ratio: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .image-preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .image-preview .placeholder {
      color: #444;
      text-align: center;
    }

    .image-preview .placeholder span {
      font-size: 2rem;
      display: block;
      margin-bottom: 0.5rem;
    }

    .image-preview .placeholder p {
      font-size: 0.8rem;
    }

    .btn-colorize {
      width: 100%;
      padding: 0.9rem;
      background: linear-gradient(135deg, #6ee7f7, #a78bfa);
      color: #000;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    .btn-colorize:hover { opacity: 0.9; }
    .btn-colorize:active { transform: scale(0.99); }
    .btn-colorize:disabled { opacity: 0.4; cursor: not-allowed; }

    .progress-wrap {
      display: none;
      margin-top: 1rem;
    }

    .progress-info {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: #888;
      margin-bottom: 6px;
    }

    .progress-track {
      background: #2a2a35;
      border-radius: 99px;
      height: 4px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #6ee7f7, #a78bfa);
      border-radius: 99px;
      width: 0%;
      transition: width 0.3s ease;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 1.25rem;
    }

    .metric {
      background: #12121a;
      border: 1px solid #2a2a35;
      border-radius: 10px;
      padding: 0.75rem;
      text-align: center;
    }

    .metric .m-label {
      font-size: 0.7rem;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }

    .metric .m-value {
      font-size: 1.2rem;
      font-weight: 600;
      color: #e8e8e8;
    }

    .btn-download {
      width: 100%;
      padding: 0.75rem;
      background: transparent;
      color: #6ee7f7;
      border: 1px solid #6ee7f7;
      border-radius: 10px;
      font-size: 0.9rem;
      cursor: pointer;
      margin-top: 0.75rem;
      transition: background 0.2s;
      display: none;
    }

    .btn-download:hover { background: rgba(110,231,247,0.1); }

    .status-badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 99px;
      font-size: 0.75rem;
      margin-left: 8px;
    }

    .badge-ready   { background: #1a3a2a; color: #4ade80; }
    .badge-none    { background: #2a2a35; color: #888; }
  </style>
</head>
<body>

<div class="container">

  <div class="header">
    <h1>🛰 Celestial</h1>
    <p>Infrared satellite image colorization using Pix2Pix deep learning
      <span class="status-badge {{ 'badge-ready' if model_loaded else 'badge-none' }}">
        {{ 'Model ready' if model_loaded else 'No model — train first' }}
      </span>
    </p>
  </div>

  <div class="card">
    <div class="grid">

      <!-- LEFT: Upload -->
      <div class="panel">
        <label>Input — infrared image</label>
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
          <span class="icon">📡</span>
          <p>Drop your IR image here</p>
          <p class="formats">.npy · .png · .jpg · .tif</p>
        </div>
        <input type="file" id="fileInput" accept=".npy,.png,.jpg,.jpeg,.tif,.tiff" style="display:none">
        <div class="image-preview" id="inputPreview" style="display:none; margin-top: 0.75rem;">
          <img id="inputImg" src="" alt="IR input">
        </div>
      </div>

      <!-- RIGHT: Output -->
      <div class="panel">
        <label>Output — colorized RGB</label>
        <div class="image-preview" id="outputPreview">
          <div class="placeholder">
            <span>🎨</span>
            <p>Result appears here</p>
          </div>
        </div>
      </div>

    </div>

    <!-- Colorize button -->
    <button class="btn-colorize" id="colorizeBtn" disabled onclick="colorize()">
      ✨ Colorize image
    </button>

    <!-- Progress bar -->
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-info">
        <span id="progressLabel">Processing...</span>
        <span id="progressPct">0%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" id="progressFill"></div>
      </div>
    </div>

    <!-- Download button -->
    <button class="btn-download" id="downloadBtn" onclick="downloadResult()">
      ⬇ Download colorized image
    </button>

    <!-- Metrics -->
    <div class="metrics">
      <div class="metric">
        <div class="m-label">PSNR</div>
        <div class="m-value" id="psnrVal">—</div>
      </div>
      <div class="metric">
        <div class="m-label">SSIM</div>
        <div class="m-value" id="ssimVal">—</div>
      </div>
      <div class="metric">
        <div class="m-label">Inference time</div>
        <div class="m-value" id="infVal">—</div>
      </div>
    </div>

  </div>
</div>

<script>
let selectedFile = null;
let resultBlob   = null;

const fileInput    = document.getElementById('fileInput');
const colorizeBtn  = document.getElementById('colorizeBtn');
const dropZone     = document.getElementById('dropZone');
const inputPreview = document.getElementById('inputPreview');
const inputImg     = document.getElementById('inputImg');
const outputPreview = document.getElementById('outputPreview');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const progressPct  = document.getElementById('progressPct');
const downloadBtn  = document.getElementById('downloadBtn');

// File selection
fileInput.addEventListener('change', function() {
  if (this.files.length > 0) handleFile(this.files[0]);
});

// Drag and drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  dropZone.querySelector('p').textContent = file.name;

  // Show preview for image files
  if (file.name.match(/\\.(png|jpg|jpeg)$/i)) {
    const reader = new FileReader();
    reader.onload = e => {
      inputImg.src = e.target.result;
      inputPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    inputPreview.style.display = 'none';
  }

  colorizeBtn.disabled = false;
}

// Progress animation
function animateProgress(steps, callback) {
  let i = 0;
  progressWrap.style.display = 'block';
  const iv = setInterval(() => {
    if (i >= steps.length) { clearInterval(iv); callback(); return; }
    progressFill.style.width = steps[i][0] + '%';
    progressLabel.textContent = steps[i][1];
    progressPct.textContent = steps[i][0] + '%';
    i++;
  }, 400);
}

// Main colorize function
async function colorize() {
  if (!selectedFile) return;

  colorizeBtn.disabled = true;
  downloadBtn.style.display = 'none';
  outputPreview.innerHTML = '<div class="placeholder"><span>⏳</span><p>Processing...</p></div>';

  const steps = [
    [20, 'Loading model...'],
    [40, 'Preprocessing IR image...'],
    [60, 'Enhancing image...'],
    [80, 'Running Pix2Pix model...'],
    [95, 'Post-processing output...'],
  ];

  animateProgress(steps, async () => {
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/colorize', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Server error');
      }

      const data = await response.json();

      // Show output image
      progressFill.style.width = '100%';
      progressPct.textContent = '100%';
      progressLabel.textContent = 'Done!';

      setTimeout(() => {
        progressWrap.style.display = 'none';

        outputPreview.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Colorized output" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;

        // Show metrics
        document.getElementById('psnrVal').textContent = data.psnr ? data.psnr + ' dB' : 'N/A';
        document.getElementById('ssimVal').textContent = data.ssim || 'N/A';
        document.getElementById('infVal').textContent  = data.inference_time + ' ms';

        // Enable download
        resultBlob = data.image;
        downloadBtn.style.display = 'block';
        colorizeBtn.disabled = false;
      }, 500);

    } catch (err) {
      progressWrap.style.display = 'none';
      outputPreview.innerHTML = `<div class="placeholder"><span>❌</span><p>${err.message}</p></div>`;
      colorizeBtn.disabled = false;
    }
  });
}

function downloadResult() {
  if (!resultBlob) return;
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + resultBlob;
  a.download = 'colorized_output.png';
  a.click();
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, model_loaded=G is not None)


@app.route("/colorize", methods=["POST"])
def colorize():
    if G is None:
        return jsonify({"error": "Model not loaded. Train first!"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    try:
        # ── Read uploaded file ──
        file_bytes = file.read()

        if filename.endswith(".npy"):
            img = np.load(io.BytesIO(file_bytes))
            if img.ndim == 3:
                img = img[:, :, 0]

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0

        elif filename.endswith((".tif", ".tiff")):
            import rasterio, tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                img = src.read(1).astype(np.float32)
            os.unlink(tmp_path)
            if img.max() > 0:
                img = img / img.max()
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # ── Preprocess ──
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img = img * 2.0 - 1.0
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        # ── Inference ──
        start = time.perf_counter()
        with torch.no_grad():
            fake_rgb = G(tensor)
        inf_time = round((time.perf_counter() - start) * 1000, 2)

        # ── Convert to image ──
        rgb_np = fake_rgb[0].cpu().numpy()
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
        rgb_np = (rgb_np + 1.0) / 2.0
        rgb_np = np.clip(rgb_np, 0, 1)
        rgb_uint8 = (rgb_np * 255).astype(np.uint8)
        rgb_bgr   = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

        # ── Encode to base64 ──
        _, buffer = cv2.imencode(".png", rgb_bgr)
        import base64
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image"          : img_b64,
            "inference_time" : inf_time,
            "psnr"           : None,
            "ssim"           : None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs/checkpoints", exist_ok=True)
    print("Starting Celestial web app...")
    print("Open your browser at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)