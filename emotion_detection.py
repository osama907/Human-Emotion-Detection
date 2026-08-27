"""Emotion detection client for the IBM Watson NLP service."""

import requests


WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for supplied text."""
    response = requests.post(
        WATSON_EMOTION_URL,
        json={"raw_document": {"text": text_to_analyze}},
        headers=WATSON_HEADERS,
        timeout=30,
    )
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
    response.raise_for_status()

    emotion = response.json()["emotionPredictions"][0]["emotion"]
    result = {
        "anger": emotion["anger"],
        "disgust": emotion["disgust"],
        "fear": emotion["fear"],
        "joy": emotion["joy"],
        "sadness": emotion["sadness"],
    }
    result["dominant_emotion"] = max(emotion, key=emotion.get)
    return result