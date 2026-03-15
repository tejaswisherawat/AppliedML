# Assignment 4 – Containerization and Continuous Integration

## Overview

In this assignment, we extended the machine learning application developed in the previous assignment by focusing on **containerization and automated testing workflows**. The main goal was to package the Flask-based spam classification application into a portable environment using Docker and ensure that the application could be reliably tested and deployed.

Additionally, we implemented automated testing and basic continuous integration practices to make sure that the application behaves correctly whenever changes are made to the codebase.

---

## Flask Application

The core application is a **Flask-based API for spam classification**.  
The API accepts a text message and returns:

- A **prediction** indicating whether the message is spam or not.
- A **propensity score**, which represents the probability that the message is spam.

The application loads a trained machine learning model from a serialized file and uses it to compute predictions through a scoring function. The API exposes a `/score` endpoint that processes incoming requests and returns the results in JSON format.

---

## Containerization Using Docker

To ensure that the application can run consistently across different environments, the Flask application was containerized using **Docker**.

A Dockerfile was created to define how the container should be built. The container setup includes:

- Selecting a Python base environment.
- Installing the necessary Python dependencies required by the application.
- Copying the application files and trained model into the container.
- Launching the Flask server when the container starts.

This approach ensures that the application can be run in an isolated environment without requiring manual installation of dependencies.

---

## Automated Testing

Testing for the application was implemented using **pytest**. The test suite includes multiple types of tests designed to verify different parts of the system.

### Unit Tests

Unit tests were written for the `score` function to verify:

- Correct output types.
- Valid prediction values.
- Probability values within the expected range.
- Proper behavior under different threshold values.

These tests ensure that the machine learning scoring logic behaves correctly.

### API Tests

Additional tests were written for the Flask application to verify:

- The homepage endpoint.
- The `/score` API endpoint when receiving JSON input.
- The `/score` API endpoint when receiving form data.
- Error handling when invalid input is provided.

These tests confirm that the API behaves correctly when interacting with users or other systems.

### Docker Integration Test

A special test was created to verify the behavior of the **containerized application**.  
This test performs the following steps:

- Builds the Docker image for the application.
- Launches the Docker container.
- Sends a request to the `/score` endpoint running inside the container.
- Verifies that the response structure is correct.
- Stops and removes the container after the test completes.

This ensures that the containerized version of the application functions correctly.

---

## Code Coverage

Code coverage analysis was performed to evaluate how much of the codebase is exercised by the tests.

Coverage reporting helps identify parts of the code that are not being tested and ensures that the application logic is well validated by the test suite. The resulting coverage report was saved for reference as part of the assignment deliverables.

---

## Continuous Integration with Git Hooks

To enforce good development practices, a **pre-commit Git hook** was implemented.

The purpose of this hook is to automatically run the test suite whenever a commit is attempted. If any test fails, the commit is aborted. This ensures that only code that passes all tests can be committed to the repository.

This mechanism acts as a simple form of continuous integration and helps maintain code reliability during development.

---

## Project Structure

The assignment directory contains the main application files, testing scripts, Docker configuration, and supporting resources.

Key components include:

- Flask application code
- Machine learning scoring logic
- Test suite implemented with pytest
- Docker configuration for containerization
- Coverage report generated from the tests
- Pre-commit hook script for automated test execution

---

## Conclusion

This assignment demonstrates how a machine learning application can be prepared for deployment by combining several important software engineering practices:

- Containerizing applications using Docker
- Writing automated tests to validate functionality
- Measuring code coverage
- Integrating automated checks using Git hooks

Together, these practices improve the reliability, portability, and maintainability of machine learning systems.