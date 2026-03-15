import pytest
import joblib
import os
import requests
import subprocess
import time

from score import score
from app import app

# Load model using relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "best_spam_model.joblib")

model = joblib.load(MODEL_PATH)


# ---------------------------
# UNIT TESTS FOR score()
# ---------------------------

def test_smoke_test():
    """Check if score function runs without crashing and returns 2 outputs."""
    result = score("Example message", model, 0.5)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_format_test():
    """Check correct output types."""
    prediction, probability = score("Example message", model, 0.7)

    assert isinstance(prediction, int)
    assert prediction in (0, 1)

    assert isinstance(probability, float)


def test_prediction_0_or_1():
    prediction, _ = score("Example message", model, 0.6)
    assert prediction in (0, 1)


def test_propensity_between_0_and_1():
    _, propensity = score("Example message", model, 0.6)
    assert 0 <= propensity <= 1


def test_threshold_0_prediction_always_1():
    prediction1, _ = score("Hello there", model, 0)
    prediction2, _ = score("Win money now!!!", model, 0)

    assert prediction1 == 1
    assert prediction2 == 1


def test_threshold_1_prediction_always_0():
    prediction1, _ = score("Hello there", model, 1)
    prediction2, _ = score("Win money now!!!", model, 1)

    assert prediction1 == 0
    assert prediction2 == 0


def test_obvious_spam():
    text = """
    Congratulations! You have won a lottery.
    Claim your prize now. Limited time offer.
    """
    prediction, _ = score(text, model, 0.5)

    assert prediction == 1


def test_obvious_non_spam():
    text = "Don't forget the meeting tomorrow at 10 AM."
    prediction, _ = score(text, model, 0.5)

    assert prediction == 0


# ---------------------------
# INTEGRATION TEST (REAL SERVER)
# ---------------------------

def test_flask_server():
    """
    Launch Flask app using subprocess,
    send request to /score endpoint,
    then terminate server.
    """

    process = subprocess.Popen(["python", "app.py"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    time.sleep(2)  # wait for server to start

    payload = {"text": "Congratulations! You have won a free prize!"}

    response = requests.post(
        "http://127.0.0.1:5000/score",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "propensity" in data

    process.terminate()


# ---------------------------
# FLASK TEST CLIENT (Cleaner)
# ---------------------------

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data


def test_score_json(client):
    response = client.post("/score",
                           json={"text": "You have won a prize!"})

    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json


def test_score_form_data(client):
    response = client.post("/score",
                           data={"text": "Win money now!"})

    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json


def test_missing_text(client):
    response = client.post("/score", json={})

    assert response.status_code == 400
    assert response.json == {"error": "No input text provided"}


# ---------------------------
# DOCKER CONTAINER TEST
# ---------------------------

def test_docker():
    """
    Build docker image, run container, test /score endpoint,
    then stop and remove the container.
    """

    image_name = "spam-flask-app"
    container_name = "spam-test-container"

    # Build docker image
    subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

    # Run docker container in background
    subprocess.run([
        "docker", "run", "-d",
        "-p", "5000:5000",
        "--name", container_name,
        image_name
    ], check=True)

    # Wait for Flask server to start
    time.sleep(5)

    try:
        payload = {"text": "free money now"}

        response = requests.post(
            "http://127.0.0.1:5000/score",
            json=payload
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "propensity" in data

        assert isinstance(data["prediction"], int)
        assert isinstance(data["propensity"], float)

    finally:
        # Stop and remove container
        subprocess.run(["docker", "stop", container_name])
        subprocess.run(["docker", "rm", container_name])


# ---------------------------
# EXTRA TESTS FOR 100% COVERAGE
# ---------------------------

def test_score_invalid_text():
    """score() should raise error if text is not string"""
    with pytest.raises(ValueError):
        score(12345, model, 0.5)


def test_score_invalid_threshold():
    """score() should raise error if threshold outside [0,1]"""
    with pytest.raises(ValueError):
        score("hello", model, 2)


def test_score_load_model_when_none():
    """score() should load model internally if model=None"""
    prediction, prob = score("Hello there", None, 0.5)

    assert isinstance(prediction, int)
    assert isinstance(prob, float)


def test_model_not_loaded(client, monkeypatch):
    """Trigger model=None branch in app.py"""
    import app as app_module

    monkeypatch.setattr(app_module, "model", None)

    response = client.post("/score", json={"text": "Hello"})

    assert response.status_code == 500
    assert response.json == {"error": "Model not loaded"}