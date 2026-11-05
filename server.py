"""Flask application for uploading and classifying emotion images."""

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from emotion_detection import (
    EmotionDetectionError,
    EmotionDetector,
    InvalidImageError,
    ModelUnavailableError,
    decode_image,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
detector = EmotionDetector()


def _analyze_upload():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        raise InvalidImageError("Choose an image to analyze.")
    return detector.predict(decode_image(uploaded_file.read()))


@app.get("/")
def index():
    """Show the image upload form."""
    return render_template("index.html")


@app.post("/predict")
def predict_page():
    """Analyze an uploaded image and render the results."""
    try:
        result = _analyze_upload()
        return render_template("index.html", result=result), 200
    except InvalidImageError as error:
        return render_template("index.html", error=str(error)), 400
    except ModelUnavailableError as error:
        return render_template("index.html", error=str(error)), 503
    except EmotionDetectionError as error:
        return render_template("index.html", error=str(error)), 500


@app.post("/api/predict")
def predict_api():
    """Return emotion scores as JSON for an uploaded image."""
    try:
        return jsonify(_analyze_upload()), 200
    except InvalidImageError as error:
        return jsonify(error=str(error)), 400
    except ModelUnavailableError as error:
        return jsonify(error=str(error)), 503
    except EmotionDetectionError as error:
        return jsonify(error=str(error)), 500


@app.get("/health")
def health():
    """Report whether a model and its labels are available."""
    status = 200 if detector.is_ready else 503
    return jsonify(status="ready" if status == 200 else "model_unavailable"), status


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    message = "Image uploads must be 8 MB or smaller."
    if request.path.startswith("/api/"):
        return jsonify(error=message), 413
    return render_template("index.html", error=message), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
