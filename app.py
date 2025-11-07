import os
import secrets
import numpy as np
from datetime import timedelta
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 🔹 Initialize Flask App
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure random key
app.permanent_session_lifetime = timedelta(minutes=30)

# 🔹 Load Pre-trained Model
MODEL_PATH = "heart_attack_risk_model.keras"
model = load_model(MODEL_PATH)

# 🔹 Configure Upload Folder
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔹 Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 🔹 Utility function to check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🔹 Image Prediction Function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return "High Risk" if prediction > 0.5 else "Low Risk"

# 🔹 Home Route
@app.route("/")
def index():
    return render_template("index.html")

# 🔹 Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        valid_users = {
            "admin": "password",
            "sarvani": "4265",
            "uma": "4223",
            "moulali": "4253",
            "abdullah": "4242"
        }

        if username in valid_users and valid_users[username] == password:
            session.permanent = True
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for('upload'))
        else:
            error = "Invalid Credentials"

    return render_template("login.html", error=error)

# 🔹 Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# 🔹 Upload Image Route
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    error = None
    if request.method == "POST":
        if 'file' not in request.files:
            error = "No file part in the request"
            return render_template("upload.html", error=error)

        f = request.files["file"]
        if f.filename == '':
            error = "No file selected"
            return render_template("upload.html", error=error)

        if not allowed_file(f.filename):
            error = "Invalid file type. Allowed types are png, jpg, jpeg, gif."
            return render_template("upload.html", error=error)

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        result = predict_image(filepath)
        
        return render_template("results.html", result=result, img_path=filepath)

    return render_template("upload.html")

# 🔹 Results Route
@app.route("/results")
def results():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    result = request.args.get("result")
    img_path = request.args.get("img_path")

    if not result or not img_path:
        return redirect(url_for("upload"))

    return render_template("results.html", result=result, img_path=img_path)

# 🔹 Analysis Route
@app.route("/analysis")
def analysis():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    stats = {
        "total_cases": 200,
        "high_risk": 80,
        "low_risk": 120,
        "accuracy": "85%"
    }
    return render_template("analysis.html", stats=stats)

# 🔹 Run the App
if __name__ == "__main__":
    app.run(debug=True)
