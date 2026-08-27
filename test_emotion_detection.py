import unittest
from unittest.mock import Mock, patch

from emotion_detection import emotion_detector


def response_for(emotion_name):
    scores = {
        "anger": 0.01,
        "disgust": 0.02,
        "fear": 0.03,
        "joy": 0.04,
        "sadness": 0.05,
    }
    scores[emotion_name] = 0.90
    return {"emotionPredictions": [{"emotion": scores}]}


class TestEmotionDetector(unittest.TestCase):
    @patch("emotion_detection.requests.post")
    def test_anger(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200, json=lambda: response_for("anger")
        )
        self.assertEqual(emotion_detector("I am furious")["dominant_emotion"], "anger")

    @patch("emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200, json=lambda: response_for("disgust")
        )
        self.assertEqual(emotion_detector("That is disgusting")["dominant_emotion"], "disgust")

    @patch("emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200, json=lambda: response_for("sadness")
        )
        self.assertEqual(emotion_detector("I feel sad")["dominant_emotion"], "sadness")

    @patch("emotion_detection.requests.post")
    def test_fear(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200, json=lambda: response_for("fear")
        )
        self.assertEqual(emotion_detector("I am afraid")["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()