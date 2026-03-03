import joblib
import os
from typing import Tuple
from sklearn.base import BaseEstimator


def score(text: str,
          model: BaseEstimator = None,
          threshold: float = 0.5) -> Tuple[int, float]:
    """
    Scores a given text using a trained sklearn classification model.

    This function:
    1. Takes raw input text.
    2. Uses a trained model (pipeline) to compute the probability
       that the text belongs to the positive class (spam).
    3. Applies a threshold to convert the probability into a binary prediction.
    
    Parameters
    ----------
    text : str
        The input text message to classify.

    model : sklearn.base.BaseEstimator, optional
        A trained sklearn model (preferably a pipeline including
        vectorizer + classifier). If not provided, the function
        loads 'best_spam_model.joblib' from the current directory.

    threshold : float, default=0.5
        A value between 0 and 1 used to convert probability into
        a binary class prediction:
            - If probability >= threshold → prediction = 1
            - Else → prediction = 0

    Returns
    -------
    prediction : int
        Binary classification result:
            1 → spam
            0 → not spam

    propensity : float
        The predicted probability (between 0 and 1)
        that the input text is spam.
    """

    # Input validation
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    # Load model if not provided
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__),
                                  "best_spam_model.joblib")
        model = joblib.load(model_path)

    # Get probability of class 1 (spam)
    propensity = model.predict_proba([text])[0][1]

    # Convert probability to binary prediction
    prediction = int(propensity >= threshold)

    return prediction, float(propensity)