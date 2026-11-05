"""Create a labeled preview grid from the training image folders."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from emotion_detection import EMOTIONS


def save_preview(
    dataset_dir: Path,
    output_path: Path = Path("reports/dataset_preview.png"),
    samples_per_class: int = 4,
) -> Path:
    """Save a grid of readable images selected from each emotion folder."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to create a dataset preview.") from error
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    figure, axes = plt.subplots(
        len(EMOTIONS),
        samples_per_class,
        figsize=(3 * samples_per_class, 3 * len(EMOTIONS)),
        squeeze=False,
    )
    try:
        for row, emotion in enumerate(EMOTIONS):
            class_dir = dataset_dir / emotion
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
            samples = []
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                try:
                    with Image.open(path) as image:
                        samples.append((path.name, image.convert("RGB").copy()))
                except (UnidentifiedImageError, OSError) as error:
                    warnings.warn(
                        f"Skipping unreadable image {path}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if len(samples) == samples_per_class:
                    break
            if not samples:
                raise ValueError(f"No readable images were found under {class_dir}.")
            for column, axis in enumerate(axes[row]):
                axis.axis("off")
                if column < len(samples):
                    name, image = samples[column]
                    axis.imshow(image)
                    axis.set_title(f"{emotion}: {name}", fontsize=9)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/Emotions Dataset/Emotions Dataset/train"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_preview.png"))
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()
    try:
        saved_path = save_preview(
            args.dataset_dir,
            args.output,
            args.samples_per_class,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved dataset preview to {saved_path}")


if __name__ == "__main__":
    main()
