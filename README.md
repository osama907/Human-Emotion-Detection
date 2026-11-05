# Image Emotion Detection

A Flask application and training project based on the notebook's three image classes: **angry**, **happy**, and **sad**. It includes selectable CNN, ResNet-34, EfficientNetB4, and Vision Transformer architectures, training and evaluation utilities, TFRecord input, image explanations, and a browser/API deployment path.

> This model classifies visible facial expression; it cannot determine a person's actual internal emotional state.

## Notebook implementation map

The original notebook is kept in [`Human_emotion_detection.ipynb`](./Human_emotion_detection.ipynb). Its reusable workflows are implemented as project code rather than depending on Colab cells:

| Notebook workflow | Project implementation |
|---|---|
| Kaggle image folders and dataset preview | `data_pipeline.py`, `preview_dataset.py` |
| Image augmentation and CutMix | `model_factory.py`, `data_pipeline.py` |
| LeNet, ResNet-34, EfficientNetB4, custom ViT, Hugging Face ViT | `model_factory.py` |
| Training, fine-tuning, checkpoints, curves, optional W&B metrics | `train.py` |
| TFRecord encoding, reading, and training | `prepare_tfrecords.py`, `data_pipeline.py` |
| Validation metrics and confusion matrix | `evaluate.py` |
| Feature maps, Grad-CAM, and ViT attention visualization | `explain.py` |
| Single-image inference and web/API app | `emotion_detection.py`, `server.py` |

Colab shell commands, interactive Kaggle/W&B login, Drive-specific weight paths, and repeated debugging cells are replaced by documented CLI options and external credential setup; secrets and datasets do not belong in the repository.

## Requirements

- Python 3.10–3.12
- The packages in `requirements.txt`
- The Kaggle dataset used by the notebook

TensorFlow's current releases do not support native Windows execution. On Windows, use WSL2 (or another supported Linux environment) for training and model inference. Keep the project and dataset inside the WSL filesystem for best performance. For the optional Hugging Face ViT, install `requirements-vit.txt`.

## Setup the dataset

The notebook uses the Kaggle dataset [`muhammadhananasghar/human-emotions-datasethes`](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes). Install the Kaggle CLI and configure your Kaggle API credentials outside this repository. Never commit `kaggle.json`.

```bash
python -m pip install -r requirements.txt
python -m pip install kaggle
kaggle datasets download -d muhammadhananasghar/human-emotions-datasethes \
  -p data --unzip
```

The training and test folders must each contain `angry/`, `happy/`, and `sad/` subfolders. The defaults match the directory layout used in the notebook:

```text
data/
└── Emotions Dataset/
    └── Emotions Dataset/
        ├── train/
        │   ├── angry/
        │   ├── happy/
        │   └── sad/
        └── test/
            ├── angry/
            ├── happy/
            └── sad/
```

If the archive extracts to another location, pass its paths explicitly.

Preview example images from each class:

```bash
python preview_dataset.py
```

## Train the model

```bash
python train.py
```

Training defaults to the notebook's EfficientNetB4 transfer-learning model and downloads ImageNet weights on the first run. The best validation checkpoint, class labels, loss/accuracy plots, and CSV history are saved under `models/` (ignored by Git). Choose an architecture and training settings with:

```bash
python train.py --model efficientnet_b4 \
  --train-dir path/to/train --validation-dir path/to/test \
  --epochs 20 --batch-size 32 --image-size 256 \
  --model-path models/emotion_model.keras
```

Available model options are `lenet`, `resnet34`, `efficientnet_b4`, `vit`, and optional `hf_vit`. Use `--no-pretrained` to initialize EfficientNet/Hugging Face backbones from scratch, `--fine-tune-layers N` to fine-tune the last N transfer-learning layers, and `--cutmix` to enable batch CutMix during training.

The notebook's optional Weights & Biases tracking is supported without an interactive login cell. Install `requirements-wandb.txt`, configure your W&B credentials outside the repository, and add `--wandb` to the training command.

## TFRecords

The notebook's TFRecord workflow is available as a reproducible preparation command:

```bash
python prepare_tfrecords.py \
  --train-dir "data/Emotions Dataset/Emotions Dataset/train" \
  --validation-dir "data/Emotions Dataset/Emotions Dataset/test" \
  --output-dir data/tfrecords
```

Train from the generated records with:

```bash
python train.py --model lenet \
  --train-tfrecord data/tfrecords/train.tfrecord \
  --validation-tfrecord data/tfrecords/validation.tfrecord
```

## Evaluate and inspect models

Generate a per-class classification report and confusion-matrix image:

```bash
python evaluate.py --model-path models/emotion_model.keras
```

Generate an image prediction explanation or inspect convolutional feature maps:

```bash
python explain.py --model-path models/emotion_model.keras --image path/to/face.jpg
python explain.py --model-path models/emotion_model.keras --image path/to/face.jpg \
  --feature-maps --output reports/features.png
python explain.py --image path/to/face.jpg --vgg16 \
  --output reports/vgg16_features.png
```

For the notebook's pretrained Hugging Face ViT attention visualization (install `requirements-vit.txt` first):

```bash
python explain.py --image path/to/face.jpg --attention \
  --output reports/vit_attention.png
```

The attention overlay uses the pretrained ViT as an exploratory visualization; it is not the trained classifier's prediction explanation. Grad-CAM is generated from the selected trained classifier.

## Run the application

```bash
python server.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and upload a PNG, JPEG, or WEBP image (up to 8 MB). Set `EMOTION_MODEL_PATH` to use a model saved elsewhere; its matching labels file must be next to it and named with the `.labels.json` suffix.

### API

Send an image as multipart form data in the `image` field:

```bash
curl -X POST -F "image=@face.jpg" http://127.0.0.1:5000/api/predict
```

The successful response contains probabilities and the dominant class:

```json
{
  "angry": 0.12,
  "happy": 0.76,
  "sad": 0.12,
  "dominant_emotion": "happy"
}
```

`GET /health` reports whether the model and labels are present. Prediction requests return `503` until a model has been trained.

## Tests

Unit and Flask route tests use a small fake model, so they do not download data or TensorFlow model weights:

```bash
python -m pytest -q
```
