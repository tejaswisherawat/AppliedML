# Assignment 4 – Docker Containerization and Continuous Integration

## Overview
In this assignment, we containerize the Flask application developed in Assignment 3 using **Docker** and implement automated testing and continuous integration using a **pre-commit Git hook**. 

The goal is to package the machine learning application in a reproducible environment and ensure that tests run automatically before committing code.

---

## Application Description
The application is a **Spam Classification API** built with Flask. It exposes an endpoint `/score` which accepts a text message and returns:

* **prediction**: binary classification (0 = not spam, 1 = spam)
* **propensity**: probability that the text is spam

The model used is a trained `sklearn` pipeline stored in:  
`best_spam_model.joblib`

---

## Docker Containerization
A Docker container was created for the Flask application. The container performs the following:

1.  Uses a **Python base image**.
2.  Installs required **dependencies**.
3.  Copies **application files** into the container.
4.  Launches the **Flask server**.

The container is defined in the `Dockerfile`.

### Build Docker Image
```bash
docker build -t spam-flask-app .