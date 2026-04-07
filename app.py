from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from keras.models import load_model

# -------------------------------
# Flask App Configuration
# -------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# Load Trained Model
# -------------------------------
try:
    model = load_model("E:/Phd_2021/final_code_1/website/models/CNN_1032026.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# -------------------------------
# ISIC 2019 Class Labels
# -------------------------------
class_labels = {
    0: "AK",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "SCC",
    7: "VASC"
}

# -------------------------------
# Disease Descriptions
# -------------------------------
disease_descriptions = {
    "AK": "Actinic keratosis is a precancerous lesion caused by long-term sun exposure.",
    "BCC": "Basal cell carcinoma is the most common type of skin cancer.",
    "BKL": "Benign keratosis includes non-cancerous lesions such as seborrheic keratosis.",
    "DF": "Dermatofibroma is a benign skin tumor often appearing as a small firm bump.",
    "MEL": "Melanoma is a dangerous form of skin cancer that develops from melanocytes. Early detection is essential.",
    "NV": "Melanocytic nevus (mole) is a common benign skin lesion.",
    "SCC": "Squamous cell carcinoma is a common skin cancer arising from squamous cells.",
    "VASC": "Vascular lesions are abnormalities of blood vessels such as angiomas or hemangiomas."
}

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded", "danger")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)

            if model is None:
                flash("Model not loaded.", "danger")
                return redirect(request.url)

            # Predict lesion class
            pred = model.predict(img_array)
            print("Prediction vector:", pred)
            pred_index = np.argmax(pred)
            pred_class = class_labels[pred_index]
            confidence = float(pred[0][pred_index]) * 100

            return render_template(
                "result.html",
                prediction=pred_class,
                confidence=round(confidence, 2),
                image_url=file_path
            )

        else:
            flash("Invalid file type. Only PNG/JPG/JPEG allowed.", "danger")
            return redirect(request.url)

    return render_template("classify.html")

@app.route("/disease/<disease_name>")
def disease_info(disease_name):
    description = disease_descriptions.get(disease_name, "No information available.")
    return render_template("disease_info.html", disease_name=disease_name, description=description)

@app.route("/skin_diseases")
def skin_diseases():
    return render_template("skin_diseases.html", diseases=disease_descriptions)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        flash("Thank you for contacting us!", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

# ==============================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
        use_reloader=False
    )