"""Image preprocessing and inference for the emotion classifier."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


EMOTIONS = ("angry", "happy", "sad")
IMAGE_SIZE = 256
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "emotion_model.keras"


class EmotionDetectionError(Exception):
    """Base error for expected image-classification failures."""


class InvalidImageError(EmotionDetectionError):
    """Raised when an upload is missing or is not a supported image."""


class ModelUnavailableError(EmotionDetectionError):
    """Raised when the trained model or its labels are unavailable."""


class PredictionError(EmotionDetectionError):
    """Raised when the model returns invalid prediction data."""


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode supported image bytes into an RGB Pillow image."""
    if not image_bytes:
        raise InvalidImageError("Choose an image to analyze.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImageError("Upload a PNG, JPEG, or WEBP image.")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error


class EmotionDetector:
    """Load a saved Keras model on demand and classify RGB images."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        image_size: int = IMAGE_SIZE,
        model: Any | None = None,
        class_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.model_path = Path(
            model_path
            or os.environ.get("EMOTION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self.image_size = image_size
        self._model = model
        self._class_names = tuple(class_names) if class_names else None
        self._load_lock = Lock()

        if image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

    @property
    def is_ready(self) -> bool:
        """Return whether model files are present without loading TensorFlow."""
        return (
            self._model is not None
            or (
                self.model_path.is_file()
                and self._labels_path.is_file()
            )
        )

    @property
    def _labels_path(self) -> Path:
        return self.model_path.with_suffix(".labels.json")

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise ModelUnavailableError(
                    f"Model not found at {self.model_path}. "
                    "Train it first with `python train.py`."
                )
            if not self._labels_path.is_file():
                raise ModelUnavailableError(
                    f"Model labels not found at {self._labels_path}. "
                    "Train the model again to generate its labels file."
                )

            try:
                labels = json.loads(self._labels_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ModelUnavailableError(
                    f"Model labels file is invalid: {self._labels_path}."
                ) from error
            image_size = self.image_size
            if isinstance(labels, dict):
                metadata = labels
                image_size = labels.get("image_size", image_size)
                labels = labels.get("classes")
                if metadata.get("model") == "hf_vit":
                    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            if (
                not isinstance(labels, list)
                or not labels
                or any(not isinstance(label, str) or not label for label in labels)
            ):
                raise ModelUnavailableError(
                    f"Model labels file must contain class names: {self._labels_path}."
                )
            if not isinstance(image_size, int) or image_size <= 0:
                raise ModelUnavailableError(
                    f"Model labels file has an invalid image_size: "
                    f"{self._labels_path}."
                )

            try:
                from tensorflow.keras.models import load_model
            except ImportError as error:
                raise ModelUnavailableError(
                    "TensorFlow is required for predictions. Install the project "
                    "requirements in a supported Python environment."
                ) from error

            self._model = load_model(self.model_path)
            self._class_names = tuple(labels)
            self.image_size = image_size

    def predict(self, image: Image.Image) -> dict[str, float | str]:
        """Return class probabilities and the highest-scoring emotion."""
        self._load_model()
        if self._class_names is None:
            self._class_names = EMOTIONS

        resized = image.convert("RGB").resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
        predictions = np.asarray(self._model.predict(batch, verbose=0))
        if (
            predictions.ndim != 2
            or predictions.shape[0] != 1
            or predictions.shape[1] != len(self._class_names)
        ):
            raise PredictionError("The model returned an unexpected number of scores.")

        scores = predictions[0]
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise PredictionError("The model returned invalid emotion scores.")
        total = float(scores.sum())
        if total <= 0:
            raise PredictionError("The model returned no usable emotion scores.")
        scores = scores / total

        result: dict[str, float | str] = {
            name: float(score) for name, score in zip(self._class_names, scores)
        }
        result["dominant_emotion"] = self._class_names[int(np.argmax(scores))]
        return result
