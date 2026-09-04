# Day 6-7: containerizes the full training pipeline --
# fetch -> validate -> build features -> retrain (champion/challenger).
#
# Distinct from the other two Dockerfiles in this repo:
#   docker/Dockerfile   - Airflow orchestration image (runs the DAG on a schedule)
#   Dockerfile.serve    - serving-only image (FastAPI over the exported model bundle)
# This one is for a single end-to-end pipeline run, e.g. to retrain from
# scratch in a clean environment without installing Python locally.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# No git binary in this image, and mlflow only uses it for informational
# run tags (commit SHA/branch) -- quiet keeps that from logging a scary
# multi-line warning on every run.
ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--synthetic"]
