"""Export the notebook's image folders to portable TFRecord files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_detection import EMOTIONS, IMAGE_SIZE
from data_pipeline import write_tfrecords


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert train/test image folders to TFRecord files."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tfrecords"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_tfrecords(
        args.train_dir, args.output_dir / "train.tfrecord", args.image_size
    )
    validation_count = write_tfrecords(
        args.validation_dir, args.output_dir / "validation.tfrecord", args.image_size
    )
    metadata = {
        "classes": list(EMOTIONS),
        "image_size": args.image_size,
        "train_examples": train_count,
        "validation_examples": validation_count,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {train_count} training and {validation_count} validation examples.")
    print(f"TFRecords saved under {args.output_dir}")


if __name__ == "__main__":
    main()
