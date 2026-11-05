"""Train, evaluate, and save one of the notebook's image classifiers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emotion_detection import DEFAULT_MODEL_PATH, EMOTIONS, IMAGE_SIZE
from model_factory import MODEL_NAMES, build_model


def train_model(
    train_dir: Path,
    validation_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    model_name: str = "efficientnet_b4",
    epochs: int = 20,
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
    cutmix: bool = False,
    fine_tune_layers: int = 0,
    pretrained: bool = True,
    train_tfrecord: Path | None = None,
    validation_tfrecord: Path | None = None,
    use_wandb: bool = False,
) -> dict[str, list[float]]:
    """Train a selected architecture and save its best checkpoint."""
    if model_name == "hf_vit":
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for training. Install requirements.txt "
            "in a supported Python environment."
        ) from error

    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {', '.join(MODEL_NAMES)}.")
    if epochs <= 0 or batch_size <= 0 or image_size <= 0:
        raise ValueError("epochs, batch_size, and image_size must be positive.")
    if fine_tune_layers < 0:
        raise ValueError("fine_tune_layers cannot be negative.")
    if bool(train_tfrecord) != bool(validation_tfrecord):
        raise ValueError("Provide both TFRecord paths or neither.")

    from data_pipeline import apply_cutmix, load_directory_datasets, load_tfrecord_dataset

    tf.keras.utils.set_random_seed(99)
    if train_tfrecord and validation_tfrecord:
        training = load_tfrecord_dataset(
            train_tfrecord, batch_size, image_size, shuffle=True
        )
        validation = load_tfrecord_dataset(
            validation_tfrecord, batch_size, image_size
        )
    else:
        training, validation = load_directory_datasets(
            train_dir, validation_dir, batch_size, image_size
        )
    if cutmix:
        training = apply_cutmix(training)

    model, backbone = build_model(
        model_name=model_name,
        image_size=image_size,
        pretrained=pretrained,
    )
    if fine_tune_layers:
        if backbone is None:
            raise ValueError(
                f"--fine-tune-layers is only available for the transfer-learning "
                f"models: efficientnet_b4 and hf_vit."
            )
        backbone.trainable = True
        if fine_tune_layers < len(backbone.layers):
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4 if fine_tune_layers else 1e-3
        ),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=min(2, len(EMOTIONS)), name="top_2_accuracy"
            ),
        ],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = model_path.parent / f"{model_path.stem}_training"
    report_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(report_dir / "history.csv")),
    ]
    wandb_run = None
    try:
        if use_wandb:
            try:
                import wandb
                from wandb.integration.keras import WandbMetricsLogger
            except ImportError as error:
                raise RuntimeError(
                    "Weights & Biases logging requires `requirements-wandb.txt`."
                ) from error
            wandb_run = wandb.init(
                project="Emotion-Detection",
                config={
                    "model": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "cutmix": cutmix,
                    "fine_tune_layers": fine_tune_layers,
                },
            )
            if wandb_run is None:
                raise RuntimeError("Weights & Biases did not create a training run.")
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        history = model.fit(
            training,
            validation_data=validation,
            epochs=epochs,
            callbacks=callbacks,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    model.save(model_path)
    model_path.with_suffix(".labels.json").write_text(
        json.dumps(
            {
                "classes": list(EMOTIONS),
                "image_size": image_size,
                "model": model_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_training_plots(history.history, report_dir)
    print(f"Saved model: {model_path}")
    print(f"Saved labels and model settings: {model_path.with_suffix('.labels.json')}")
    print(f"Saved training history and plots: {report_dir}")
    return history.history


def write_training_plots(history: dict[str, list[float]], output_dir: Path) -> None:
    """Save loss and accuracy curves without making plotting a training dependency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required to save training plots. Install project requirements."
        ) from error

    for metric, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
        if metric not in history or f"val_{metric}" not in history:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history[metric], label=f"Training {metric}")
        axis.plot(history[f"val_{metric}"], label=f"Validation {metric}")
        axis.set(title=f"Model {title}", xlabel="Epoch", ylabel=title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}.png", dpi=150)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one of the image emotion classifiers."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="efficientnet_b4",
        help="lenet, resnet34, efficientnet_b4, vit, or optional Hugging Face hf_vit.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/Emotions Dataset/Emotions Dataset/train"),
        help="Training folder containing angry/, happy/, and sad/ subfolders.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("data/Emotions Dataset/Emotions Dataset/test"),
        help="Validation folder containing angry/, happy/, and sad/ subfolders.",
    )
    parser.add_argument(
        "--train-tfrecord",
        type=Path,
        help="Use a prepared training TFRecord instead of --train-dir.",
    )
    parser.add_argument(
        "--validation-tfrecord",
        type=Path,
        help="Use a prepared validation TFRecord instead of --validation-dir.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--fine-tune-layers", type=int, default=0)
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize transfer-learning backbones without pretrained weights.",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        parser.error("--epochs, --batch-size, and --image-size must be positive.")
    if args.fine_tune_layers < 0:
        parser.error("--fine-tune-layers cannot be negative.")
    if bool(args.train_tfrecord) != bool(args.validation_tfrecord):
        parser.error("Provide both --train-tfrecord and --validation-tfrecord.")
    if args.model in {"vit", "hf_vit"} and args.image_size % 16:
        parser.error("ViT image size must be divisible by 16.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    train_model(
        train_dir=arguments.train_dir,
        validation_dir=arguments.validation_dir,
        model_path=arguments.model_path,
        model_name=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        image_size=arguments.image_size,
        cutmix=arguments.cutmix,
        fine_tune_layers=arguments.fine_tune_layers,
        pretrained=not arguments.no_pretrained,
        train_tfrecord=arguments.train_tfrecord,
        validation_tfrecord=arguments.validation_tfrecord,
        use_wandb=arguments.wandb,
    )
