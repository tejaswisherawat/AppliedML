FROM python:3.12

WORKDIR /app

COPY app.py .
COPY score.py .
COPY best_spam_model.joblib .

RUN pip install flask joblib scikit-learn numpy

CMD ["python", "app.py"]