
import sys
import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Make src/ importable ──────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

import pid as pid_module
import model as model_manager
import torch

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB upload limit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# ── Load model once at startup ────────────────────────────────────────────────
print(f"[INIT] Loading model on {DEVICE} …")
_model, _, _, _ = model_manager.get_model(DEVICE)
_model.eval()
print("[INIT] Model ready.")


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def residual_to_b64(residual: np.ndarray) -> str:
    """
    Amplify the PiD residual ×8 so it's visible, then encode as PNG.
    residual values are raw float differences (not 0-255 normalised).
    """
    amp = np.clip(residual * 8 + .5, 0, 1)*255
    amp=amp.astype(np.uint8)
    img = Image.fromarray(amp)
    return pil_to_b64(img)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file or not allowed(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    try:
        # ── 1. Load image ──────────────────────────────────────────────────
        pil_img = Image.open(file.stream).convert("RGB")
        w_orig, h_orig = pil_img.size
        display_img = pil_img.resize((224, 224))

        # ── 2. PiD residual ────────────────────────────────────────────────
        residual = pid_module.apply_pid_algorithm(pil_img)   # (224,224,3) float32

        # ── 3. Build tensor for model ──────────────────────────────────────
        res_tensor = (
            torch.from_numpy(residual)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ── 4. Inference ───────────────────────────────────────────────────
        with torch.no_grad():
            outputs = _model(res_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        predicted = int(np.argmax(probs))
        label = "AI-Generated" if predicted == 1 else "Real Photo"
        confidence = float(probs[predicted]) * 100
        prob_real = float(probs[0]) * 100
        prob_fake = float(probs[1]) * 100

        # ── 5. Residual statistics ─────────────────────────────────────────
        abs_res = np.abs(residual)
        stats = {
            "mean":  round(float(abs_res.mean()), 6),
            "std":   round(float(abs_res.std()),  6),
            "max":   round(float(abs_res.max()),  4),
            "min":   round(float(abs_res.min()),  4),
        }

        # ── 6. Encode images for response ──────────────────────────────────
        original_b64 = pil_to_b64(display_img)
        residual_b64 = residual_to_b64(residual)

        return jsonify({
            "label":        label,
            "is_ai":        predicted == 1,
            "confidence":   round(confidence, 2),
            "prob_real":    round(prob_real, 2),
            "prob_fake":    round(prob_fake, 2),
            "stats":        stats,
            "original_b64": original_b64,
            "residual_b64": residual_b64,
            "image_info": {
                "width":  w_orig,
                "height": h_orig,
                "device": str(DEVICE),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
