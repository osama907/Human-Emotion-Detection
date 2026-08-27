"""Flask web interface for the emotion detector."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and analyze submitted text."""
    result = None
    error = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            error = "Please enter text to analyze."
        else:
            result = emotion_detector(text)
            if result is None:
                error = "The emotion service could not analyze that text."
    return render_template("index.html", result=result, error=error, text=text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
