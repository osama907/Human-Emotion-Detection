# Emotion Detection

A Flask application that sends text to the IBM Watson NLP Emotion Predictions service and displays scores for anger, disgust, fear, joy, sadness, and the dominant emotion.

## Run locally

```powershell
python -m pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` in a browser.

## Test and analyze

```powershell
python -m pytest -q
python -m pylint server.py
```

Set `WATSON_EMOTION_URL` to point at a compatible Watson NLP endpoint when using a different service URL.