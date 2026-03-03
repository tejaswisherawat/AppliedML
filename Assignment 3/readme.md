# Assignment 3 — Spam Classifier Deployment & Testing

## 🎯 Objective

The goal of this assignment was to take a trained spam classification model and convert it into a reliable, testable, and deployable system. Instead of stopping at model training, we focused on building a complete ML engineering workflow.

## 🔄 What We Did

1. **Trained and Saved the Model**
   - Built and selected the best spam classifier using sklearn.
   - Saved the trained model using `joblib` for reuse.

2. **Created an Inference Layer (`score.py`)**
   - Implemented a `score()` function that:
     - Accepts raw text
     - Computes spam probability
     - Applies a configurable threshold
     - Returns prediction (0/1) and probability

3. **Wrote Unit Tests (`test.py`)**
   - Verified correct output format and behavior
   - Tested edge cases (threshold = 0 and 1)
   - Ensured probability bounds (0 to 1)
   - Checked obvious spam and non-spam cases

4. **Built a Flask API (`app.py`)**
   - Created a `/score` endpoint
   - Accepted POST requests with text input
   - Returned prediction and propensity in JSON format

5. **Implemented Integration Testing**
   - Launched Flask server using subprocess
   - Sent HTTP requests to localhost
   - Validated responses
   - Terminated server after testing

6. **Generated Test Coverage Report**
   - Used `pytest --cov` to measure how much of the code was tested
   - Achieved 91% overall coverage

## ✅ Final Outcome

We transformed a trained ML model into a production-style system with:
- Clean inference abstraction
- Automated testing
- API deployment
- Measurable test coverage

This assignment demonstrated a complete ML deployment pipeline beyond notebook experimentation.