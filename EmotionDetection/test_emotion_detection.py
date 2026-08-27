from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


EMOTION_RESPONSE = {
    "emotionPredictions": [
        {
            "emotion": {
                "anger": 0.01,
                "disgust": 0.02,
                "fear": 0.03,
                "joy": 0.90,
                "sadness": 0.04,
            }
        }
    ]
}


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_returns_scores(mock_post):
    mock_post.return_value = Mock(status_code=200, json=lambda: EMOTION_RESPONSE)
    result = emotion_detector("I am very happy")

    assert result["anger"] == 0.01
    assert result["disgust"] == 0.02
    assert result["fear"] == 0.03
    assert result["joy"] == 0.90
    assert result["sadness"] == 0.04
    assert result["dominant_emotion"] == "joy"


@patch("EmotionDetection.emotion_detection.requests.post")
def test_emotion_detector_handles_bad_request(mock_post):
    mock_post.return_value = Mock(status_code=400)

    assert emotion_detector("") is None