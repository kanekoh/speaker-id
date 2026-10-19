import os
import re
import hashlib
import shutil
from flask import Flask, request, jsonify, send_from_directory, abort, send_file
import sherpa_onnx
import numpy as np
from scipy import signal as scipy_signal
import json
import soundfile as sf
import time
import logging
import sys
import zipfile
import io
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from werkzeug.exceptions import HTTPException


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.info  # ショートカット用

app = Flask(__name__)
app.logger.handlers = []
app.logger.propagate = True

# 話者識別モデルの初期化
MODEL_PATH = os.getenv("SPEAKER_MODEL_PATH", "/app/model/model.onnx")
SAMPLE_RATE = 16000
# コサイン類似度がこの値未満なら「該当者なし」とする（実データで調整すること）
SCORE_THRESHOLD = float(os.getenv("SPEAKER_SCORE_THRESHOLD", "0.5"))

# 識別結果の履歴（スコア確認用、メモリ上のみで再起動すると消える）
identify_log = deque(maxlen=int(os.getenv("IDENTIFY_LOG_SIZE", "200")))

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

def get_embedding_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """任意フォーマット（webm/ogg/wav等）の音声バイト列から話者埋め込みベクトルを抽出する"""
    audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buf = io.BytesIO()
    audio_seg.export(wav_buf, format="wav")
    wav_buf.seek(0)
    audio, sr = sf.read(wav_buf, dtype='float32')
    return get_embedding_from_array(audio, sr)

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
    embedding = np.array(extractor.compute(stream), dtype=np.float32)
    # sherpa-onnx の埋め込みは未正規化なので、内積=コサイン類似度になるよう L2 正規化する
    return embedding / (np.linalg.norm(embedding) + 1e-9)

def get_embedding_from_file(filepath: str) -> np.ndarray:
    """WAVファイルから話者埋め込みベクトルを抽出する"""
    audio, sr = sf.read(filepath, dtype='float32')
    return get_embedding_from_array(audio, sr)

# ディレクトリの初期化
# profiles/<name>/<sample_id>.wav に 1 人あたり複数の登録音声を置く
PROFILE_DIR = os.getenv("SPEAKER_PROFILE_DIR", "profiles")
META_FILE = os.path.join(PROFILE_DIR, "metadata.json")
os.makedirs(PROFILE_DIR, exist_ok=True)

# 識別名はディレクトリ名に使うので英数字・_・- のみ許可する
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAMPLE_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{3}$")


def _model_tag() -> str:
    """モデルファイルの識別子（埋め込みキャッシュの無効化に使う）"""
    try:
        h = hashlib.sha1()
        with open(MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "nomodel"


MODEL_TAG = _model_tag()

# メタ情報読み込み
if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

samples = {}   # name -> {sample_id: 正規化済み埋め込み}
profiles = {}  # name -> 全サンプルの平均（正規化済み）


def person_dir(name: str) -> str:
    return os.path.join(PROFILE_DIR, name)


def new_sample_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_metadata():
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def centroid(embeddings) -> np.ndarray:
    mean = np.mean(np.stack(list(embeddings)), axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def load_sample_embedding(wav_path: str) -> np.ndarray:
    """登録音声の埋め込みを返す。計算済みの .npy があればそれを使う"""
    cache_path = f"{wav_path[:-4]}.{MODEL_TAG}.npy"
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(wav_path):
        return np.load(cache_path)
    embedding = get_embedding_from_file(wav_path)
    np.save(cache_path, embedding)
    return embedding


def remove_sample_files(name: str, sample_id: str):
    """サンプルの wav と埋め込みキャッシュ（.npy）を削除する"""
    for filename in os.listdir(person_dir(name)):
        if filename.startswith(sample_id + "."):
            os.remove(os.path.join(person_dir(name), filename))


def refresh_profile(name: str):
    """サンプルの平均からプロファイルを作り直す。サンプルが無ければ削除"""
    if samples.get(name):
        profiles[name] = centroid(samples[name].values())
    else:
        samples.pop(name, None)
        profiles.pop(name, None)


def migrate_legacy_profiles():
    """旧形式 profiles/<name>.wav（1人1本）を profiles/<name>/<sample_id>.wav に移す"""
    for filename in sorted(os.listdir(PROFILE_DIR)):
        legacy_path = os.path.join(PROFILE_DIR, filename)
        if not (filename.endswith(".wav") and os.path.isfile(legacy_path)):
            continue
        name = filename[:-4]
        if not NAME_RE.match(name):
            logging.warning(f"[migrate] skip '{filename}': invalid name")
            continue
        try:
            os.makedirs(person_dir(name), exist_ok=True)
            mtime = datetime.fromtimestamp(os.path.getmtime(legacy_path))
            dest = os.path.join(person_dir(name), mtime.strftime("%Y%m%d-%H%M%S-000") + ".wav")
            # 中身が WebM などの場合もあるので pydub で本物の WAV に変換する
            AudioSegment.from_file(legacy_path).export(dest, format="wav")
            os.remove(legacy_path)
            logging.info(f"[migrate] '{filename}' -> '{dest}'")
        except Exception as e:
            logging.error(f"[migrate] could not migrate '{filename}': {e}")


def load_profiles():
    """profiles/ から全登録者のサンプルを読み込み、プロファイルを作る"""
    global samples, profiles
    migrate_legacy_profiles()
    samples, profiles = {}, {}
    for name in sorted(os.listdir(PROFILE_DIR)):
        if not (NAME_RE.match(name) and os.path.isdir(person_dir(name))):
            continue
        for filename in sorted(os.listdir(person_dir(name))):
            if not filename.endswith(".wav"):
                continue
            try:
                samples.setdefault(name, {})[filename[:-4]] = \
                    load_sample_embedding(os.path.join(person_dir(name), filename))
            except Exception as e:
                logging.error(f"[startup] Could not load '{name}/{filename}': {e}")
        refresh_profile(name)
    log(f"[startup] loaded {len(profiles)} speakers, "
        f"{sum(len(v) for v in samples.values())} samples (model {MODEL_TAG})")


load_profiles()

@app.errorhandler(Exception)
def handle_exception(e):
    # abort(401) などの HTTP 例外はそのままのステータスで返す
    if isinstance(e, HTTPException):
        return e
    print(f"[unhandled exception] {type(e).__name__}: {e}", flush=True)
    logging.exception(f"[unhandled] {e}")
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/test.html")
def test():
    return send_from_directory(".", "test.html")

@app.route("/scores.html")
def scores_page():
    return send_from_directory(".", "scores.html")

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
    global metadata
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    # 旧形式のバックアップもここで新形式に移行される
    load_profiles()
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
    if not NAME_RE.match(name):
        return jsonify({"error": "識別名は半角英数字・_・- のみ使えます"}), 400

    os.makedirs(person_dir(name), exist_ok=True)
    sample_id = new_sample_id()
    save_path = os.path.join(person_dir(name), f"{sample_id}.wav")

    try:
        raw_bytes = file.read()
        audio_seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
        audio_seg.export(save_path, format="wav")
    except Exception as e:
        logging.exception("[register] audio conversion failed")
        return jsonify({"error": f"音声変換失敗: {e}"}), 500

    try:
        t0 = time.time()
        embedding = load_sample_embedding(save_path)
        print(f"[register] get_embedding took {time.time() - t0:.3f} sec")
    except Exception as e:
        logging.exception("[register] get_embedding failed")
        os.remove(save_path)
        return jsonify({"error": str(e)}), 500

    # 既存サンプルとの一致度（低ければ録り直しの目安）
    existing = samples.get(name, {})
    consistency = round(float(np.dot(embedding, centroid(existing.values()))), 3) if existing else None

    samples.setdefault(name, {})[sample_id] = embedding
    refresh_profile(name)

    if kana or name not in metadata:
        metadata[name] = {"kana": kana or name}
    save_metadata()

    print(f"[register] Total time: {time.time() - start_time:.3f} sec")
    return jsonify({
        "status": "ok",
        "name": name,
        "kana": metadata[name]["kana"],
        "sample_id": sample_id,
        "num_samples": len(samples[name]),
        "consistency": consistency,
    })


@app.route("/profiles", methods=["GET"])
def list_profiles():
    """登録者とサンプルの一覧。各サンプルの「他のサンプルとの一致度」も返す"""
    require_auth()
    people = []
    for name in sorted(samples):
        items = []
        for sample_id, emb in samples[name].items():
            others = [e for sid, e in samples[name].items() if sid != sample_id]
            items.append({
                "id": sample_id,
                "consistency": round(float(np.dot(emb, centroid(others))), 3) if others else None,
            })
        people.append({
            "name": name,
            "kana": metadata.get(name, {}).get("kana", name),
            "samples": items,
        })
    return jsonify({"people": people})


@app.route("/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    require_auth()
    if not NAME_RE.match(name) or name not in samples:
        return jsonify({"error": "not found"}), 404
    shutil.rmtree(person_dir(name), ignore_errors=True)
    samples.pop(name, None)
    refresh_profile(name)
    metadata.pop(name, None)
    save_metadata()
    return jsonify({"status": "deleted", "name": name})


@app.route("/profiles/<name>/samples/<sample_id>", methods=["DELETE"])
def delete_sample(name, sample_id):
    require_auth()
    if (not NAME_RE.match(name) or not SAMPLE_ID_RE.match(sample_id)
            or sample_id not in samples.get(name, {})):
        return jsonify({"error": "not found"}), 404
    remove_sample_files(name, sample_id)
    del samples[name][sample_id]
    refresh_profile(name)
    if name not in samples:
        shutil.rmtree(person_dir(name), ignore_errors=True)
        metadata.pop(name, None)
        save_metadata()
    return jsonify({"status": "deleted", "name": name, "sample_id": sample_id,
                    "num_samples": len(samples.get(name, {}))})

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
        embedding = get_embedding_from_bytes(audio_bytes)
        log(f"[identify] get_embedding took {time.time() - t0:.3f} sec")
    except Exception as e:
        logging.exception("[identify] get_embedding failed")
        return jsonify({"error": str(e)}), 500

    t0 = time.time()
    scores = {name: float(np.dot(embedding, prof_embedding))
              for name, prof_embedding in profiles.items()}
    best_name = max(scores, key=scores.get) if scores else None
    best_score = scores[best_name] if best_name else -1
    log(f"[identify] similarity search took {time.time() - t0:.3f} sec")

    accepted = best_name is not None and best_score >= SCORE_THRESHOLD
    identify_log.append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "source": request.form.get("source", "api"),
        "result": best_name if accepted else None,
        "candidate": best_name,
        "score": round(best_score, 3),
        "scores": {name: round(score, 3) for name, score in scores.items()},
    })
    score_list = [
        {"name": name, "kana": metadata.get(name, {}).get("kana", name), "score": round(score, 3)}
        for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]

    log(f"[identify] best={best_name} score={best_score:.3f} threshold={SCORE_THRESHOLD}")
    log(f"[identify] Total time: {time.time() - start_time:.3f} sec")

    if not accepted:
        return jsonify({
            "name": None,
            "kana": None,
            "score": round(best_score, 3),
            "candidate": best_name,
            "threshold": SCORE_THRESHOLD,
            "scores": score_list,
        })

    kana = metadata.get(best_name, {}).get("kana", best_name)
    return jsonify({
        "name": best_name,
        "kana": kana,
        "score": round(best_score, 3),
        "threshold": SCORE_THRESHOLD,
        "scores": score_list,
    })

@app.route("/identify_log", methods=["GET"])
def get_identify_log():
    require_auth()
    return jsonify({
        "threshold": SCORE_THRESHOLD,
        "people": [{"name": name, "kana": metadata.get(name, {}).get("kana", name)}
                   for name in sorted(profiles)],
        "entries": list(identify_log),
    })

@app.route("/identify_log", methods=["DELETE"])
def clear_identify_log():
    require_auth()
    identify_log.clear()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
