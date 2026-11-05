"""Evaluate a saved classifier and write a confusion-matrix report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from emotion_detection import EMOTIONS, IMAGE_SIZE


def evaluate_model(
    model_path: Path,
    validation_dir: Path,
    output_dir: Path = Path("reports"),
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
) -> dict[str, object]:
    """Evaluate every validation image and save JSON and chart reports."""
    labels_path = model_path.with_suffix(".labels.json")
    class_names = EMOTIONS
    if labels_path.is_file():
        metadata = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            image_size = metadata.get("image_size", image_size)
            classes = metadata.get("classes")
            if (
                not isinstance(classes, list)
                or not classes
                or any(not isinstance(name, str) or not name for name in classes)
            ):
                raise ValueError(f"Invalid class names in {labels_path}.")
            class_names = tuple(classes)
            if metadata.get("model") == "hf_vit":
                os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        elif (
            isinstance(metadata, list)
            and metadata
            and all(isinstance(name, str) and name for name in metadata)
        ):
            class_names = tuple(metadata)
        else:
            raise ValueError(f"Invalid labels metadata in {labels_path}.")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    try:
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow, scikit-learn, and Matplotlib are required for evaluation."
        ) from error

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=False,
    )
    model = tf.keras.models.load_model(model_path)
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        truths.append(labels.numpy())
        predictions.append(np.argmax(probabilities, axis=1))
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "classes": list(class_names),
        "samples": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(title="Emotion classification", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(f"Accuracy: {report_data['accuracy']:.4f}")
    print(f"Saved evaluation report to {output_dir}")
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("data/Emotions Dataset/Emotions Dataset/test"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.image_size <= 0:
        parser.error("--batch-size and --image-size must be positive.")
    evaluate_model(
        args.model_path,
        args.validation_dir,
        args.output_dir,
        args.batch_size,
        args.image_size,
    )


if __name__ == "__main__":
    main()
