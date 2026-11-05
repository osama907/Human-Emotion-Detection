"""Keras model builders for the architectures explored in the notebook."""

from __future__ import annotations

from typing import Any

from emotion_detection import EMOTIONS, IMAGE_SIZE


MODEL_NAMES = ("lenet", "resnet34", "efficientnet_b4", "vit", "hf_vit")


def build_model(
    model_name: str = "efficientnet_b4",
    image_size: int = IMAGE_SIZE,
    pretrained: bool = True,
) -> tuple[Any, Any | None]:
    """Build a classifier and return its optional fine-tunable backbone."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for model creation. Install requirements.txt "
            "in a supported Python environment."
        ) from error

    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model {model_name!r}; choose from {', '.join(MODEL_NAMES)}."
        )
    if image_size <= 0 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16.")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    augmented = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.025),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )(inputs)

    if model_name == "lenet":
        features = tf.keras.layers.Rescaling(1.0 / 255)(augmented)
        regularizer = tf.keras.regularizers.L2(0.0)
        features = tf.keras.layers.Conv2D(
            6, 3, activation="relu", kernel_regularizer=regularizer
        )(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(features)
        features = tf.keras.layers.Dropout(0.0)(features)
        features = tf.keras.layers.Conv2D(
            16, 3, activation="relu", kernel_regularizer=regularizer
        )(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(features)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dense(
            1024, activation="relu", kernel_regularizer=regularizer
        )(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.Dropout(0.0)(features)
        features = tf.keras.layers.Dense(
            128, activation="relu", kernel_regularizer=regularizer
        )(features)
        features = tf.keras.layers.BatchNormalization()(features)
        outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(features)
        return tf.keras.Model(inputs, outputs, name="lenet_emotion_classifier"), None

    if model_name == "resnet34":
        features = tf.keras.layers.Conv2D(
            64, 7, strides=2, padding="same", activation="relu"
        )(augmented)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.MaxPooling2D(3, strides=2)(features)
        for filters, blocks, first_stride in (
            (64, 3, 1),
            (128, 4, 2),
            (256, 6, 2),
            (512, 3, 2),
        ):
            for block in range(blocks):
                stride = first_stride if block == 0 else 1
                shortcut = features
                features = tf.keras.layers.Conv2D(
                    filters,
                    3,
                    strides=stride,
                    padding="same",
                    activation="relu",
                )(features)
                features = tf.keras.layers.BatchNormalization()(features)
                features = tf.keras.layers.Conv2D(
                    filters, 3, padding="same", activation="relu"
                )(features)
                features = tf.keras.layers.BatchNormalization()(features)
                if stride != 1 or shortcut.shape[-1] != filters:
                    shortcut = tf.keras.layers.Conv2D(
                        filters, 1, strides=stride, activation="relu"
                    )(shortcut)
                    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
                features = tf.keras.layers.Add()([features, shortcut])
                features = tf.keras.layers.Activation("relu")(features)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(features)
        return tf.keras.Model(inputs, outputs, name="resnet34_emotion_classifier"), None

    if model_name == "vit":
        features = tf.keras.layers.Resizing(image_size, image_size)(augmented)
        features = tf.keras.layers.Rescaling(1.0 / 255)(features)
        patch_size = 16
        patch_count = (image_size // patch_size) ** 2
        patches = tf.keras.layers.Conv2D(
            768, patch_size, strides=patch_size, padding="valid"
        )(features)
        patches = tf.keras.layers.Reshape((patch_count, 768))(patches)
        positions = tf.range(start=0, limit=patch_count, delta=1)
        encoded = patches + tf.keras.layers.Embedding(patch_count, 768)(positions)
        for _ in range(2):
            residual = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=192
            )(residual, residual)
            encoded = tf.keras.layers.Add()([encoded, attention])
            residual = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
            residual = tf.keras.layers.Dense(768, activation="gelu")(residual)
            residual = tf.keras.layers.Dense(768, activation="gelu")(residual)
            encoded = tf.keras.layers.Add()([encoded, residual])
        features = tf.keras.layers.Flatten()(encoded)
        features = tf.keras.layers.Dense(128, activation="gelu")(features)
        features = tf.keras.layers.Dense(128, activation="gelu")(features)
        outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(features)
        return tf.keras.Model(inputs, outputs, name="vision_transformer_emotion"), None

    if model_name == "hf_vit":
        try:
            from transformers import TFViTModel, ViTConfig
        except ImportError as error:
            raise RuntimeError(
                "The Hugging Face ViT requires the optional "
                "`transformers` dependency. Install it with "
                "`python -m pip install -r requirements-vit.txt`."
            ) from error
        backbone = (
            TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            if pretrained
            else TFViTModel(ViTConfig())
        )
        backbone.trainable = False
        resized = tf.keras.layers.Resizing(224, 224)(augmented)
        normalized = tf.keras.layers.Rescaling(1.0 / 255)(resized)
        encoded = backbone.vit(normalized)[0][:, 0, :]
        outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(encoded)
        return tf.keras.Model(inputs, outputs, name="huggingface_vit_emotion"), backbone

    backbone_name = "EfficientNetB4"
    application = tf.keras.applications.EfficientNetB4
    backbone = application(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    features = backbone(augmented)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(1024, activation="relu")(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Dense(128, activation="relu")(features)
    outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(features)
    return (
        tf.keras.Model(inputs, outputs, name=f"{backbone_name.lower()}_emotion"),
        backbone,
    )
