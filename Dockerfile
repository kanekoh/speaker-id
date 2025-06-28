FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg build-essential libsndfile1 && \
    pip install --no-cache-dir resemblyzer flask numpy scipy

WORKDIR /app
COPY . /app

CMD ["python", "-u", "app.py"]

