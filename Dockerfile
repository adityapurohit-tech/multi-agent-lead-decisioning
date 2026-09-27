FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY seeds ./seeds
COPY scripts ./scripts
COPY run.py .
# Train at build time so the image ships with a model and model card.
RUN python -m app.ml.train
EXPOSE 5000
CMD ["python", "run.py"]
