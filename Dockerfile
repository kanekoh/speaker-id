FROM python:3.12-slim

WORKDIR /app

# sherpa-onnx モデルのURL（ビルド時に上書き可能）
# モデル一覧: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recog-models
ARG SPEAKER_MODEL_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

RUN apt-get update && apt-get install -y \
    wget \
    libsndfile1 \
    ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 話者識別モデルをダウンロード
RUN mkdir -p /app/model && \
    wget -q -O /app/model/model.onnx "${SPEAKER_MODEL_URL}"

COPY . /app

CMD ["python", "-u", "app.py"]
