"""Generate Grad-CAM overlays and convolutional feature-map grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from emotion_detection import EMOTIONS, IMAGE_SIZE, decode_image


def _nested_layers(layer: Any):
    for child in getattr(layer, "layers", ()):
        yield child
        yield from _nested_layers(child)


def _model_image_size(model_path: Path) -> int:
    labels_path = model_path.with_suffix(".labels.json")
    if not labels_path.is_file():
        return IMAGE_SIZE
    metadata = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(metadata, dict):
        if metadata.get("model") == "hf_vit":
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        size = metadata.get("image_size", IMAGE_SIZE)
        if isinstance(size, int) and size > 0:
            return size
    return IMAGE_SIZE


def _find_layer(model, layer_name: str | None, layer_type):
    matches = [
        layer
        for layer in _nested_layers(model)
        if isinstance(layer, layer_type)
    ]
    if layer_name:
        for layer in matches:
            if layer.name == layer_name:
                return layer
        raise ValueError(f"No {layer_type.__name__} named {layer_name!r} in model.")
    if not matches:
        raise ValueError(f"Model has no {layer_type.__name__} layers.")
    return matches[-1]


def _gradcam_outputs(model, target_layer, image_batch):
    import tensorflow as tf

    top_level = {layer.name: layer for layer in model.layers}
    if target_layer.name in top_level:
        grad_model = tf.keras.Model(
            model.inputs,
            [target_layer.output, model.output],
        )
        return grad_model(image_batch, training=False)

    containing_model = next(
        (
            layer
            for layer in model.layers
            if target_layer in tuple(_nested_layers(layer))
            and isinstance(layer, tf.keras.Model)
        ),
        None,
    )
    if containing_model is None:
        raise ValueError("Could not locate the convolution's containing model.")

    dual_output_backbone = tf.keras.Model(
        containing_model.input,
        [target_layer.output, containing_model.output],
    )
    augmentation = top_level.get("data_augmentation")
    features_input = (
        augmentation(image_batch, training=False)
        if augmentation is not None
        else image_batch
    )
    activations, features = dual_output_backbone(features_input, training=False)
    index = model.layers.index(containing_model)
    for layer in model.layers[index + 1 :]:
        features = layer(features, training=False)
    return activations, features


def make_gradcam(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    layer_name: str | None = None,
) -> tuple[str, Path]:
    """Write a Grad-CAM overlay and return its predicted class and path."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    image_size = _model_image_size(model_path)
    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow and Matplotlib are required to create Grad-CAM visualizations."
        ) from error
    original = decode_image(image_path.read_bytes())
    resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
    model = tf.keras.models.load_model(model_path)
    target_layer = _find_layer(model, layer_name, tf.keras.layers.Conv2D)

    with tf.GradientTape() as tape:
        activations, probabilities = _gradcam_outputs(model, target_layer, image_batch)
        class_index = tf.argmax(probabilities[0])
        class_score = probabilities[:, class_index]
    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError(
            "Gradients are unavailable for this layer. Select a connected "
            "convolutional layer with --layer."
        )

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation = activations[0]
    heatmap = tf.reduce_sum(activation * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) <= 0:
        raise ValueError("The selected layer produced an empty Grad-CAM heatmap.")
    heatmap = heatmap / maximum
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (original.height, original.width),
        method="bilinear",
    )[..., 0]
    color_map = plt.get_cmap("jet")
    colors = color_map(heatmap.numpy())[..., :3]
    source = np.asarray(original, dtype=np.float32) / 255.0
    overlay = np.clip(source * 0.55 + colors * 0.45, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, overlay)

    metadata_path = model_path.with_suffix(".labels.json")
    class_names = EMOTIONS
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and isinstance(metadata.get("classes"), list):
            class_names = tuple(metadata["classes"])
    return class_names[int(class_index.numpy())], output_path


def save_feature_maps(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    max_channels: int = 16,
) -> Path:
    """Save a grid of channels from the model's last convolutional layer."""
    image_size = _model_image_size(model_path)
    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow and Matplotlib are required to visualize feature maps."
        ) from error

    image_size = _model_image_size(model_path)
    image = decode_image(image_path.read_bytes()).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    batch = np.asarray(image, dtype=np.float32)[np.newaxis, ...]
    model = tf.keras.models.load_model(model_path)
    target_layer = _find_layer(model, None, tf.keras.layers.Conv2D)
    feature_maps, _ = _gradcam_outputs(model, target_layer, batch)
    features = feature_maps[0]
    count = min(int(features.shape[-1]), max_channels)
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for index in range(count):
        axes[index].imshow(features[..., index], cmap="viridis")
        axes[index].set_title(f"Feature {index + 1}")
        axes[index].axis("off")
    for axis in axes[count:]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_vgg16_feature_maps(
    image_path: Path,
    output_path: Path,
    image_size: int = IMAGE_SIZE,
    max_layers: int = 13,
) -> Path:
    """Recreate the notebook's untrained VGG16 convolution feature-map view."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if image_size <= 0 or max_layers <= 0:
        raise ValueError("image_size and max_layers must be positive.")
    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow and Matplotlib are required to visualize VGG16 features."
        ) from error

    image = decode_image(image_path.read_bytes()).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    batch = np.asarray(image, dtype=np.float32)[np.newaxis, ...]
    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=(image_size, image_size, 3),
    )
    convolution_layers = [
        layer
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ][-max_layers:]
    feature_model = tf.keras.Model(
        backbone.input,
        [layer.output for layer in convolution_layers],
    )
    feature_maps = feature_model(batch, training=False)

    columns = 4
    rows = len(convolution_layers)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 2.8 * rows),
        squeeze=False,
    )
    for row, (layer, activation) in enumerate(zip(convolution_layers, feature_maps)):
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column < int(activation.shape[-1]):
                axis.imshow(activation[0, ..., column], cmap="viridis")
                if column == 0:
                    axis.set_title(layer.name, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_huggingface_attention(
    image_path: Path,
    output_path: Path,
) -> Path:
    """Render the pretrained Hugging Face ViT's final CLS attention map."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from transformers import TFViTModel, ViTConfig
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow, Matplotlib, and the optional transformers dependency "
            "are required for ViT attention maps. Install requirements-vit.txt."
        ) from error
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    original = decode_image(image_path.read_bytes())
    resized = original.resize((224, 224), Image.Resampling.BILINEAR)
    batch = np.asarray(resized, dtype=np.float32)[np.newaxis, ...] / 255.0
    config = ViTConfig(output_attentions=True)
    model = TFViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=config,
    )
    outputs = model(batch, output_attentions=True, training=False)
    last_attention = outputs.attentions[-1]
    cls_attention = tf.reduce_mean(last_attention[:, :, 0, 1:], axis=1)
    patch_grid = int(np.sqrt(int(cls_attention.shape[-1])))
    if patch_grid * patch_grid != int(cls_attention.shape[-1]):
        raise ValueError("The ViT returned an unexpected patch attention shape.")
    heatmap = tf.reshape(cls_attention[0], (patch_grid, patch_grid))
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (original.height, original.width)
    )[..., 0]
    colors = plt.get_cmap("jet")(heatmap.numpy())[..., :3]
    source = np.asarray(original, dtype=np.float32) / 255.0
    overlay = np.clip(source * 0.55 + colors * 0.45, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, overlay)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/gradcam.png"))
    parser.add_argument("--layer", help="Convolutional layer name for Grad-CAM.")
    parser.add_argument(
        "--feature-maps",
        action="store_true",
        help="Save a feature-map grid instead of a Grad-CAM overlay.",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Save the pretrained Hugging Face ViT attention overlay.",
    )
    parser.add_argument(
        "--vgg16",
        action="store_true",
        help="Save the notebook's untrained VGG16 feature-map visualization.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max-layers", type=int, default=13)
    args = parser.parse_args()
    selected_modes = sum((args.attention, args.feature_maps, args.vgg16))
    if selected_modes > 1:
        parser.error("--attention, --feature-maps, and --vgg16 are exclusive.")
    if args.image_size <= 0 or args.max_layers <= 0:
        parser.error("--image-size and --max-layers must be positive.")
    if args.vgg16:
        saved_path = save_vgg16_feature_maps(
            args.image,
            args.output,
            image_size=args.image_size,
            max_layers=args.max_layers,
        )
        print(f"Saved VGG16 feature maps to {saved_path}")
        return
    if args.attention:
        saved_path = save_huggingface_attention(args.image, args.output)
        print(f"Saved pretrained ViT attention map to {saved_path}")
        return
    if args.model_path is None:
        parser.error("--model-path is required for Grad-CAM and model feature maps.")
    if args.feature_maps:
        saved_path = save_feature_maps(
            args.model_path, args.image, args.output
        )
        print(f"Saved feature maps to {saved_path}")
    else:
        predicted, saved_path = make_gradcam(
            args.model_path, args.image, args.output, args.layer
        )
        print(f"Predicted emotion: {predicted}")
        print(f"Saved Grad-CAM overlay to {saved_path}")


if __name__ == "__main__":
    main()
