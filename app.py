import os
import traceback
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

# =========================
# TENSORFLOW & CUDA SETUP
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# LOAD MODEL (ABSOLUTE PATH)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No image file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Unsupported file format. Please upload a valid image (JPG, JPEG, PNG, WEBP, BMP)."
            }), 400

        # =========================
        # IMAGE PROCESSING (IN-MEMORY)
        # =========================
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "Failed to read image. Please ensure the file is a valid image."}), 400

        # Pillow compatibility for bilinear resampling filter
        resample_filter = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        img = img.resize((224, 224), resample_filter)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDICTION
        # =========================
        raw_pred = model.predict(img_array, verbose=0)[0][0]
        prediction = float(raw_pred)

        if prediction > 0.5:
            result = "Pneumonia Detected"
            result_class = "danger"
            confidence = prediction * 100.0
            recommendation = (
                "Please consult a healthcare professional immediately "
                "for proper clinical diagnosis and treatment guidance."
            )
        else:
            result = "Normal"
            result_class = "success"
            confidence = (1.0 - prediction) * 100.0
            recommendation = (
                "Your chest X-ray appears normal. "
                "Always consult a medical professional for official confirmation."
            )

        return jsonify({
            "result": result,
            "result_class": result_class,
            "confidence": f"{confidence:.2f}%",
            "recommendation": recommendation
        })

    except Exception as e:
        print("ANALYZE ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": "An error occurred during prediction. Please try again."}), 500


# =========================
# LOCAL DEV ONLY
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
