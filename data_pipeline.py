"""Directory and TFRecord input pipelines for model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emotion_detection import EMOTIONS, IMAGE_SIZE


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for dataset preparation and training."
        ) from error
    return tf


def load_directory_datasets(
    train_dir: Path,
    validation_dir: Path,
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
) -> tuple[Any, Any]:
    """Load notebook-style class directories as batched one-hot datasets."""
    tf = _tensorflow()
    for directory in (train_dir, validation_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        observed = {path.name for path in directory.iterdir() if path.is_dir()}
        expected = set(EMOTIONS)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"{directory} must contain exactly {', '.join(EMOTIONS)} "
                f"class folders; missing={missing}, extra={extra}."
            )

    options = {
        "labels": "inferred",
        "label_mode": "categorical",
        "class_names": list(EMOTIONS),
        "color_mode": "rgb",
        "batch_size": batch_size,
        "image_size": (image_size, image_size),
        "seed": 99,
    }
    training = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        **options,
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        shuffle=False,
        **options,
    )
    autotune = tf.data.AUTOTUNE
    return (
        training.prefetch(autotune),
        validation.prefetch(autotune),
    )


def apply_cutmix(dataset: Any) -> Any:
    """Apply batch CutMix with area-corrected soft labels."""
    tf = _tensorflow()

    def mix(images, labels):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        permutation = tf.random.shuffle(tf.range(batch_size))
        paired_images = tf.gather(images, permutation)
        paired_labels = tf.gather(labels, permutation)

        samples = tf.random.gamma([2], alpha=2.0)
        lam = samples[0] / tf.reduce_sum(samples)
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_height = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
        cut_width = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
        center_y = tf.random.uniform([], 0, height, dtype=tf.int32)
        center_x = tf.random.uniform([], 0, width, dtype=tf.int32)
        y1 = tf.maximum(center_y - cut_height // 2, 0)
        y2 = tf.minimum(center_y + cut_height // 2, height)
        x1 = tf.maximum(center_x - cut_width // 2, 0)
        x2 = tf.minimum(center_x + cut_width // 2, width)

        rows = tf.range(height)[:, tf.newaxis]
        columns = tf.range(width)[tf.newaxis, :]
        patch_mask = tf.logical_and(
            tf.logical_and(rows >= y1, rows < y2),
            tf.logical_and(columns >= x1, columns < x2),
        )
        patch_mask = patch_mask[tf.newaxis, :, :, tf.newaxis]
        mixed_images = tf.where(patch_mask, paired_images, images)
        patch_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        keep_ratio = 1.0 - patch_area / tf.cast(height * width, tf.float32)
        mixed_labels = keep_ratio * labels + (1.0 - keep_ratio) * paired_labels
        return mixed_images, mixed_labels

    return dataset.map(mix, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )


def write_tfrecords(
    source_dir: Path,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
) -> int:
    """Convert class-organized images into one portable TFRecord file."""
    tf = _tensorflow()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for label_index, emotion in enumerate(EMOTIONS):
            class_dir = source_dir / emotion
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                image = tf.io.read_file(str(path))
                decoded = tf.io.decode_image(
                    image, channels=3, expand_animations=False
                )
                resized = tf.image.resize(decoded, (image_size, image_size))
                encoded = tf.io.encode_jpeg(tf.cast(resized, tf.uint8)).numpy()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[encoded])
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label_index])
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())
                count += 1
    if count == 0:
        raise ValueError(f"No supported images were found under {source_dir}.")
    return count


def load_tfrecord_dataset(
    record_path: Path,
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
    shuffle: bool = False,
) -> Any:
    """Read examples created by write_tfrecords as one-hot batches."""
    tf = _tensorflow()
    if not record_path.is_file():
        raise FileNotFoundError(f"TFRecord file does not exist: {record_path}")
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse(example):
        fields = tf.io.parse_single_example(example, feature_spec)
        image = tf.io.decode_jpeg(fields["image"], channels=3)
        image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(fields["label"], depth=len(EMOTIONS))
        return image, label

    dataset = tf.data.TFRecordDataset([str(record_path)]).map(
        parse, num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(1024, seed=99)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
