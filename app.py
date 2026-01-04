import os

# =========================
# RENDER + TENSORFLOW SAFETY
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================
# STANDARD IMPORTS
# =========================
from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
from PIL import Image
from tensorflow import keras

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =========================
# LOAD MODEL (ABSOLUTE PATH)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.keras")

model = keras.models.load_model(MODEL_PATH)

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
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # =========================
        # IMAGE PROCESSING (NO DISK LOAD)
        # =========================
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction > 0.5:
            result = "Pneumonia Detected"
            result_class = "danger"
            confidence = prediction * 100
            recommendation = (
                "Please consult a healthcare professional immediately "
                "for proper diagnosis and treatment."
            )
        else:
            result = "Normal"
            result_class = "success"
            confidence = (1 - prediction) * 100
            recommendation = (
                "Your chest X-ray appears normal. "
                "Always consult a medical professional for confirmation."
            )

        return jsonify({
            "result": result,
            "result_class": result_class,
            "confidence": f"{confidence:.2f}%",
            "recommendation": recommendation
        })

    except Exception as e:
        import traceback
        print("ANALYZE ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================
# LOCAL DEV ONLY
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
