import os
from flask import Flask, request, jsonify, send_from_directory, abort, send_file
import sherpa_onnx
import numpy as np
from scipy import signal as scipy_signal
import json
import soundfile as sf
from io import BytesIO
import time
import logging
import zipfile
import io
from pydub import AudioSegment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.info  # ショートカット用

app = Flask(__name__)

# 話者識別モデルの初期化
MODEL_PATH = os.getenv("SPEAKER_MODEL_PATH", "/app/model/model.onnx")
SAMPLE_RATE = 16000

extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=MODEL_PATH,
    num_threads=2,
)
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

# 認証トークン
API_TOKEN = os.environ.get("SPEAKERID_API_KEY")
if not API_TOKEN:
    raise RuntimeError(
        "環境変数 SPEAKERID_API_KEY が設定されていません。"
        "Render等の環境変数に必ずセットしてください！"
    )
print(f"SPEAKERID_API_KEY loaded: {API_TOKEN[:6]}...（{len(API_TOKEN)}文字）")


def check_auth():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth.split(" ", 1)[1]
    return token == API_TOKEN

def require_auth():
    if not check_auth():
        abort(401, description="Unauthorized")

def get_embedding_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """音声配列から話者埋め込みベクトルを抽出する"""
    # モノラル化
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # 16kHz にリサンプリング
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy_signal.resample(audio, num_samples)
    audio = audio.astype(np.float32)
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    stream.input_finished()
    return np.array(extractor.compute(stream))

def get_embedding_from_file(filepath: str) -> np.ndarray:
    """WAVファイルから話者埋め込みベクトルを抽出する"""
    audio, sr = sf.read(filepath, dtype='float32')
    return get_embedding_from_array(audio, sr)

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

# 音声埋め込み読み込み（壊れたファイルはpydubで修復してから読む）
profiles = {}
for filename in os.listdir(PROFILE_DIR):
    if filename.endswith(".wav"):
        filepath = os.path.join(PROFILE_DIR, filename)
        name = filename.rsplit(".", 1)[0]
        try:
            profiles[name] = get_embedding_from_file(filepath)
        except Exception as e:
            logging.warning(f"[startup] Failed to load profile '{filename}': {e} — attempting conversion")
            try:
                audio_seg = AudioSegment.from_file(filepath)
                audio_seg.export(filepath, format="wav")
                profiles[name] = get_embedding_from_file(filepath)
                logging.info(f"[startup] Converted and loaded '{filename}' successfully")
            except Exception as e2:
                logging.error(f"[startup] Could not recover '{filename}': {e2}")

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[unhandled exception] {type(e).__name__}: {e}", flush=True)
    logging.exception(f"[unhandled] {e}")
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/test.html")
def test():
    return send_from_directory(".", "test.html")

@app.route("/check_token", methods=["GET"])
def check_token_endpoint():
    require_auth()
    return jsonify({"ok": True})

@app.route("/backup", methods=["GET"])
def backup():
    require_auth()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(PROFILE_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, PROFILE_DIR)
                zf.write(filepath, arcname)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip",
        as_attachment=True, download_name="profiles_backup.zip"
    )

@app.route("/restore", methods=["POST"])
def restore():
    require_auth()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    buf = io.BytesIO(file.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(PROFILE_DIR)
    global metadata, profiles
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    profiles = {}
    for filename in os.listdir(PROFILE_DIR):
        if filename.endswith(".wav"):
            name = filename.rsplit(".", 1)[0]
            profiles[name] = get_embedding_from_file(os.path.join(PROFILE_DIR, filename))
    return jsonify({"status": "restored"})

@app.route("/register", methods=["POST"])
def register():
    print("[register] ---- request received ----", flush=True)
    require_auth()
    print("[register] auth passed", flush=True)
    start_time = time.time()

    if "audio" not in request.files or "name" not in request.form:
        return jsonify({"error": "audio and name are required"}), 400

    file = request.files["audio"]
    name = request.form["name"].strip()
    kana = request.form.get("kana", "").strip()

    if not name:
        return jsonify({"error": "Name is empty"}), 400

    save_path = os.path.join(PROFILE_DIR, f"{name}.wav")

    try:
        raw_bytes = file.read()
        audio_seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
        audio_seg.export(save_path, format="wav")
    except Exception as e:
        logging.exception("[register] audio conversion failed")
        return jsonify({"error": f"音声変換失敗: {e}"}), 500

    try:
        t0 = time.time()
        profiles[name] = get_embedding_from_file(save_path)
        print(f"[register] get_embedding took {time.time() - t0:.3f} sec")
    except Exception as e:
        logging.exception("[register] get_embedding failed")
        return jsonify({"error": str(e)}), 500

    metadata[name] = {"kana": kana}
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[register] Total time: {time.time() - start_time:.3f} sec")
    return jsonify({"status": "ok", "name": name, "kana": kana})

@app.route("/identify", methods=["POST"])
def identify():
    require_auth()
    start_time = time.time()
    log("[identify] request received")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    audio_bytes = file.read()

    try:
        t0 = time.time()
        audio, sr = sf.read(BytesIO(audio_bytes), dtype='float32')
        embedding = get_embedding_from_array(audio, sr)
        log(f"[identify] get_embedding took {time.time() - t0:.3f} sec")
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    app.run(host="0.0.0.0", port=8080, debug=False)
