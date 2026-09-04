# Deployment notes

## Current deployment status

No verified live deployment, Docker configuration, API service, or database was present in the supplied project. This repository is prepared as a Streamlit application.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, select the repository, branch `main`, and entry point `app.py`.
3. Deploy without secrets; the current application does not require environment variables.

## Operational constraints

- YOLO, EasyOCR, and video processing may exceed free-host resource or execution limits.
- Uploaded and generated files are stored on the local instance and should be treated as temporary.
- The configured upload limit is 500 MB, but the hosting provider may enforce a lower practical limit.
- The application processes a whole video synchronously and has no job queue, authentication, monitoring service, or durable storage.

For reliable production use, add an asynchronous processing architecture, authenticated object storage, structured logging, monitoring, retention controls, and calibrated detection logic. These are recommendations, not current features.

