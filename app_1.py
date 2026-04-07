from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# ==============================
# Flask App Configuration
# ==============================

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# Load Models
# ==============================

print("Loading models...")

try:
    skin_tone_model = load_model(
        "D:/Skin_disease/final_code_1/website/models/skin_disease_model_transfer_learning.h5"
    )

    skin_disease_model = load_model(
        "D:/Skin_disease/final_code_1/website/models/skin_disease_model_combain.h5"
    )

    print("Models loaded successfully!")

except Exception as e:
    print("Error loading models:", e)
    skin_tone_model = None
    skin_disease_model = None


# ==============================
# Labels
# ==============================

skin_tone_labels = {
    0: "Dark",
    1: "Light",
    2: "Medium",
    3: "Olive"
}

skin_disease_labels = {
    0: "BA-Cellulitis",
    1: "BA-Impetigo",
    2: "FU-Athlete's Foot",
    3: "FU-Nail Fungus",
    4: "FU-Ringworm",
    5: "PA-Cutaneous Larva Migrans",
    6: "VI-Chickenpox",
    7: "VI-Shingles"
}

# ==============================
# Disease Descriptions
# ==============================

disease_descriptions = {

    "BA-Cellulitis":
        "Cellulitis is a bacterial infection of the skin and underlying tissues causing redness and swelling.",

    "BA-Impetigo":
        "Impetigo is a contagious bacterial skin infection commonly affecting children.",

    "FU-Athlete's Foot":
        "Athlete's foot is a fungal infection causing itching and cracked skin on the feet.",

    "FU-Nail Fungus":
        "Nail fungus causes nails to become thick, brittle, and discolored.",

    "FU-Ringworm":
        "Ringworm is a fungal infection producing circular rashes on the skin.",

    "PA-Cutaneous Larva Migrans":
        "A parasitic skin infection causing winding itchy tracks.",

    "VI-Chickenpox":
        "Chickenpox is a viral infection causing itchy red spots.",

    "VI-Shingles":
        "Shingles causes painful rashes due to reactivation of the varicella-zoster virus."
}

# ==============================
# Skin Detection Function
# ==============================

def detect_skin(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None, "Failed to load image."

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]

    percentage = (skin_pixels / total_pixels) * 100

    if percentage < 30:
        return None, f"Only {percentage:.2f}% skin detected. Please upload clearer skin image."

    return image_path, None


# ==============================
# Image Preprocessing
# ==============================

def preprocess_image(path):

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


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
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        if file and file.filename.lower().endswith((".png", ".jpg", ".jpeg")):

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(file_path)

            # Detect skin
            skin_path, error = detect_skin(file_path)

            if error:
                flash(error, "error")
                return redirect(request.url)

            img = preprocess_image(skin_path)

            if skin_tone_model and skin_disease_model:

                tone_pred = skin_tone_model.predict(img)
                disease_pred = skin_disease_model.predict(img)

                tone_index = np.argmax(tone_pred)
                disease_index = np.argmax(disease_pred)

                tone_result = skin_tone_labels[tone_index]
                disease_result = skin_disease_labels[disease_index]

                tone_conf = tone_pred[0][tone_index] * 100
                disease_conf = disease_pred[0][disease_index] * 100

                return render_template(
                    "result.html",
                    skin_tone_result=tone_result,
                    skin_disease_result=disease_result,
                    skin_tone_confidence=tone_conf,
                    skin_disease_confidence=disease_conf,
                    image_url=file_path
                )

            else:
                flash("Models not loaded.", "error")
                return redirect(request.url)

        else:
            flash("Invalid file type.", "error")
            return redirect(request.url)

    return render_template("classify.html")


@app.route("/disease/<disease_name>")
def disease_info(disease_name):

    description = disease_descriptions.get(
        disease_name,
        "No information available."
    )

    return render_template(
        "disease_info.html",
        disease_name=disease_name,
        description=description
    )


@app.route("/skin_diseases")
def skin_diseases():

    return render_template(
        "skin_diseases.html",
        diseases=disease_descriptions
    )


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
# Run App
# ==============================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
        use_reloader=False
    )