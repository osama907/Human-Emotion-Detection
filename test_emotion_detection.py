from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from emotion_detection import (
    EmotionDetector,
    InvalidImageError,
    PredictionError,
    decode_image,
)
from model_factory import MODEL_NAMES


class FakeModel:
    def __init__(self, scores):
        self.scores = np.asarray([scores], dtype=np.float32)
        self.batch_shape = None

    def predict(self, batch, verbose=0):
        self.batch_shape = batch.shape
        return self.scores


def image_bytes(image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (20, 12), color=(200, 80, 60)).save(buffer, format=image_format)
    return buffer.getvalue()


def test_decode_image_converts_supported_image_to_rgb():
    image = decode_image(image_bytes())

    assert image.mode == "RGB"
    assert image.size == (20, 12)


@pytest.mark.parametrize("contents", [b"", b"not an image"])
def test_decode_image_rejects_empty_or_invalid_bytes(contents):
    with pytest.raises(InvalidImageError):
        decode_image(contents)


def test_detector_returns_scores_and_dominant_emotion():
    model = FakeModel([0.2, 0.7, 0.1])
    detector = EmotionDetector(model=model)

    result = detector.predict(decode_image(image_bytes()))

    assert result == {
        "angry": pytest.approx(0.2),
        "happy": pytest.approx(0.7),
        "sad": pytest.approx(0.1),
        "dominant_emotion": "happy",
    }
    assert model.batch_shape == (1, 256, 256, 3)


def test_detector_rejects_invalid_model_scores():
    detector = EmotionDetector(model=FakeModel([0.5, float("nan"), 0.5]))

    with pytest.raises(PredictionError, match="invalid emotion scores"):
        detector.predict(decode_image(image_bytes()))


def test_detector_rejects_model_output_with_wrong_class_count():
    detector = EmotionDetector(model=FakeModel([0.5, 0.5]))

    with pytest.raises(PredictionError, match="unexpected number of scores"):
        detector.predict(decode_image(image_bytes()))


def test_notebook_model_architectures_are_available():
    assert MODEL_NAMES == ("lenet", "resnet34", "efficientnet_b4", "vit", "hf_vit")
