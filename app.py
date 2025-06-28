# speaker_id/app.py
from flask import Flask, request, jsonify, send_from_directory
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import os
import json
import soundfile as sf
from io import BytesIO
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.info  # ショートカット用

app = Flask(__name__)
encoder = VoiceEncoder()

# ディレクトリの初期化
PROFILE_DIR = "profiles"
META_FILE = os.path.join(PROFILE_DIR, "metadata.json")
os.makedirs(PROFILE_DIR, exist_ok=True)

# メタ情報読み込み
if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

# 音声埋め込み読み込み
profiles = {}
for filename in os.listdir(PROFILE_DIR):
    if filename.endswith(".wav"):
        name = filename.rsplit(".", 1)[0]
        wav = preprocess_wav(os.path.join(PROFILE_DIR, filename))
        profiles[name] = encoder.embed_utterance(wav)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/test.html")
def test():
    return send_from_directory(".", "test.html")

@app.route("/register", methods=["POST"])
def register():
    start_time = time.time()  # 開始時間

    if "audio" not in request.files or "name" not in request.form:
        return jsonify({"error": "audio and name are required"}), 400

    file = request.files["audio"]
    name = request.form["name"].strip()
    kana = request.form.get("kana", "").strip()

    if not name:
        return jsonify({"error": "Name is empty"}), 400

    save_path = os.path.join(PROFILE_DIR, f"{name}.wav")
    file.save(save_path)

    try:
        t0 = time.time()
        wav = preprocess_wav(save_path)
        print(f"[register] preprocess_wav took {time.time() - t0:.3f} sec")

        t0 = time.time()
        profiles[name] = encoder.embed_utterance(wav)
        print(f"[register] embed_utterance took {time.time() - t0:.3f} sec")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    metadata[name] = {"kana": kana}
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[register] Total time: {time.time() - start_time:.3f} sec")
    return jsonify({"status": "ok", "name": name, "kana": kana})

@app.route("/identify", methods=["POST"])
def identify():
    start_time = time.time()
    log("[identify] request received")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    log("[identify] request.files読み込み開始")
    file = request.files["audio"]
    log("[identify] request.files読み込み完了")
    audio_bytes = file.read()
    log("[identify] file.read() 完了")

    try:
        t0 = time.time()
        wav_np, sr = sf.read(BytesIO(audio_bytes))         # ✅ この行が必要！
        wav = preprocess_wav(wav_np, source_sr=sr)         # ✅ NumPy配列＋SRを渡す
        log(f"[identify] preprocess_wav took {time.time() - t0:.3f} sec")

        t0 = time.time()
        embedding = encoder.embed_utterance(wav)
        log(f"[identify] embed_utterance took {time.time() - t0:.3f} sec")

    except Exception as e:
        import traceback
        traceback.print_exc()  # ← 追加すると詳細な例外ログが出ます
        return jsonify({"error": str(e)}), 500

    best_name = None
    best_score = -1

    t0 = time.time()
    for name, prof_embedding in profiles.items():
        score = float(np.dot(embedding, prof_embedding))
        if score > best_score:
            best_score = score
            best_name = name
    log(f"[identify] similarity search took {time.time() - t0:.3f} sec")

    kana = metadata.get(best_name, {}).get("kana", best_name)

    log(f"[identify] Total time: {time.time() - start_time:.3f} sec")
    return jsonify({
        "name": best_name,
        "kana": kana,
        "score": round(best_score, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
