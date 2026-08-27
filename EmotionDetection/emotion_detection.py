"""Emotion detection client for the IBM Watson NLP service."""

import os

import requests


WATSON_EMOTION_URL = os.getenv(
    "WATSON_EMOTION_URL",
    "https://sn-watson-emotion.labs.coursera.org/emotion/predict",
)


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for supplied text."""
    response = requests.post(
        WATSON_EMOTION_URL,
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )
    if response.status_code == 400:
        return None
    response.raise_for_status()

    emotion = response.json()["emotionPredictions"][0]["emotion"]
    scores = {
        "anger": emotion["anger"],
        "disgust": emotion["disgust"],
        "fear": emotion["fear"],
        "joy": emotion["joy"],
        "sadness": emotion["sadness"],
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores