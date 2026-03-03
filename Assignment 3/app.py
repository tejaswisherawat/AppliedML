from flask import Flask, request, jsonify
import joblib
import os
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model using relative path (portable & safe)
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "best_spam_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route('/')
def home():
    """
    Simple homepage with HTML form.
    """
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spam Classifier</title>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form action="/score" method="post">
                <label for="text">Enter Text:</label><br>
                <input type="text" id="text" name="text" required><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
    """


@app.route('/score', methods=['POST'])
def score_api():
    """
    API endpoint that:
    - Accepts POST request with text (JSON or form-data)
    - Returns JSON containing:
        prediction (0 or 1)
        propensity (probability between 0 and 1)
    """

    try:
        # Ensure model loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Handle JSON input
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
        else:
            # Handle form-data
            text = request.form.get("text", "").strip()

        # Validate input
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Call scoring function
        prediction, probability = score(text, model, 0.55)

        return jsonify({
            "prediction": int(prediction),
            "propensity": float(probability)
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)