from io import BytesIO
from unittest.mock import patch

from PIL import Image

import server
from emotion_detection import ModelUnavailableError


def valid_png():
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color="blue").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_homepage_renders_upload_form():
    response = server.app.test_client().get("/")

    assert response.status_code == 200
    assert b"Emotion detector" in response.data
    assert b"enctype=\"multipart/form-data\"" in response.data


def test_api_returns_prediction_for_uploaded_image():
    result = {
        "angry": 0.1,
        "happy": 0.8,
        "sad": 0.1,
        "dominant_emotion": "happy",
    }
    with patch.object(server.detector, "predict", return_value=result):
        response = server.app.test_client().post(
            "/api/predict",
            data={"image": (valid_png(), "face.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert response.get_json() == result


def test_api_requires_a_file():
    response = server.app.test_client().post("/api/predict")

    assert response.status_code == 400
    assert response.get_json()["error"] == "Choose an image to analyze."


def test_api_rejects_non_image_upload():
    response = server.app.test_client().post(
        "/api/predict",
        data={"image": (BytesIO(b"not an image"), "face.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "valid image" in response.get_json()["error"]


def test_api_reports_missing_model():
    with patch.object(
        server.detector,
        "predict",
        side_effect=ModelUnavailableError("Model is not trained."),
    ):
        response = server.app.test_client().post(
            "/api/predict",
            data={"image": (valid_png(), "face.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 503
    assert response.get_json() == {"error": "Model is not trained."}
