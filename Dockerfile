FROM python:3.10-slim

# システム依存パッケージのみインストール
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    libsndfile1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.3.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 依存関係ファイルを先にコピー（キャッシュ利用効率化）
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# コード・必要なファイルだけコピー
COPY . /app

CMD ["python", "-u", "app.py"]

