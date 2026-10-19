import os
import re
import math
import subprocess
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

# 埋め込み計算のスレッド数（ラズパイ 4/5 は 4 コア）
NUM_THREADS = int(os.getenv("SPEAKER_NUM_THREADS", str(min(4, os.cpu_count() or 1))))

extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=MODEL_PATH,
    num_threads=NUM_THREADS,
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

# 無音除去・長さチェックの設定
FRAME_SEC = 0.03            # 音量を判定するフレーム長
SILENCE_FLOOR_DB = -50.0    # これより小さい音は常に無音扱い（dBFS）
SILENCE_RANGE_DB = 30.0     # 最も大きいフレームからこれ以上小さいフレームを無音扱い
SPEECH_PAD_FRAMES = 5       # 声の前後に残す余白（フレーム数、5 × 30ms = 150ms）
MIN_REGISTER_SEC = float(os.getenv("SPEAKER_MIN_REGISTER_SEC", "3.0"))
MIN_IDENTIFY_SEC = float(os.getenv("SPEAKER_MIN_IDENTIFY_SEC", "1.0"))
# 識別に使う声の最大長（埋め込みの計算時間は長さに比例するため。0 で無制限）
MAX_IDENTIFY_SEC = float(os.getenv("SPEAKER_MAX_IDENTIFY_SEC", "6.0"))
# 前処理を変えたら上げる（保存済みの埋め込みキャッシュを作り直させるため）
PREPROCESS_VERSION = "vad1"


def decode_audio_bytes(audio_bytes: bytes):
    """任意フォーマット（wav/webm/ogg等）の音声バイト列を (波形, サンプルレート) にする"""
    # WAV など libsndfile が読める形式は直接読む（ffmpeg の起動は重いので避ける）
    try:
        return sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        pass
    # Chrome の MediaRecorder（WebM/Opus）などは ffmpeg で 16kHz モノラルに一度で変換する
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=audio_bytes, capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise ValueError(f"音声を読み込めません: {result.stderr.decode(errors='replace').strip()[-200:]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), SAMPLE_RATE


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """フレームごとの音量で無音区間を取り除き、声のある部分だけをつなげて返す"""
    frame = int(SAMPLE_RATE * FRAME_SEC)
    n = len(audio) // frame
    if n == 0:
        return audio[:0]
    frames = audio[:n * frame].reshape(n, frame)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    threshold = max(SILENCE_FLOOR_DB, rms_db.max() - SILENCE_RANGE_DB)
    voiced = rms_db > threshold
    # 語頭・語尾が切れないよう、声のあるフレームの前後も残す
    kernel = np.ones(2 * SPEECH_PAD_FRAMES + 1)
    voiced = np.convolve(voiced, kernel, mode="same") > 0
    return frames[voiced].reshape(-1)


def preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """モノラル化・16kHz へのリサンプリング・無音除去をした波形を返す"""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # FFT 方式（resample）は長さによって極端に遅くなるので、多相フィルタ方式を使う
        g = math.gcd(SAMPLE_RATE, sr)
        audio = scipy_signal.resample_poly(audio, SAMPLE_RATE // g, sr // g)
    return trim_silence(audio.astype(np.float32))


def speech_seconds(speech: np.ndarray) -> float:
    return len(speech) / SAMPLE_RATE


def embed(speech: np.ndarray) -> np.ndarray:
    """前処理済みの波形から話者埋め込みベクトルを抽出する"""
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, speech)
    stream.input_finished()
    embedding = np.array(extractor.compute(stream), dtype=np.float32)
    # sherpa-onnx の埋め込みは未正規化なので、内積=コサイン類似度になるよう L2 正規化する
    return embedding / (np.linalg.norm(embedding) + 1e-9)


def get_embedding_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """音声配列から話者埋め込みベクトルを抽出する"""
    return embed(preprocess(audio, sr))


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
# 埋め込みキャッシュ（.npy）のファイル名に入れる識別子
EMBED_TAG = f"{MODEL_TAG}-{PREPROCESS_VERSION}"

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


def cache_path_for(wav_path: str) -> str:
    return f"{wav_path[:-4]}.{EMBED_TAG}.npy"


def load_sample_embedding(wav_path: str) -> np.ndarray:
    """登録音声の埋め込みを返す。計算済みの .npy があればそれを使う"""
    cache_path = cache_path_for(wav_path)
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
            # 中身が WebM などの場合もあるので本物の WAV に変換する
            with open(legacy_path, "rb") as f:
                audio, sr = decode_audio_bytes(f.read())
            sf.write(dest, audio, sr)
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
            if filename.endswith(".npy") and not filename.endswith(f".{EMBED_TAG}.npy"):
                # モデルや前処理が変わって使われなくなったキャッシュを消す
                os.remove(os.path.join(person_dir(name), filename))
                continue
            if not filename.endswith(".wav"):
                continue
            try:
                samples.setdefault(name, {})[filename[:-4]] = \
                    load_sample_embedding(os.path.join(person_dir(name), filename))
            except Exception as e:
                logging.error(f"[startup] Could not load '{name}/{filename}': {e}")
        refresh_profile(name)
    log(f"[startup] loaded {len(profiles)} speakers, "
        f"{sum(len(v) for v in samples.values())} samples (embedding {EMBED_TAG})")


load_profiles()

# 初回の識別だけ遅くならないよう、起動時に一度推論しておく
_t = time.perf_counter()
embed(np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1)
log(f"[startup] warmup took {time.perf_counter() - _t:.3f} sec (threads={NUM_THREADS})")

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
    start_time = time.perf_counter()

    if "audio" not in request.files or "name" not in request.form:
        return jsonify({"error": "audio and name are required"}), 400

    file = request.files["audio"]
    name = request.form["name"].strip()
    kana = request.form.get("kana", "").strip()

    if not name:
        return jsonify({"error": "Name is empty"}), 400
    if not NAME_RE.match(name):
        return jsonify({"error": "識別名は半角英数字・_・- のみ使えます"}), 400
    # 打ち間違いで別人として登録したり、他人の録音に混ぜたりしないための確認
    # mode を省略した場合は従来どおり（あれば追加、なければ新規）
    mode = request.form.get("mode", "")
    if mode == "new" and name in samples:
        return jsonify({"error": f"識別名「{name}」はすでに登録されています。"
                                 "この人に録音を足す場合は「登録済みの人に追加」を選んでください"}), 409
    if mode == "add" and name not in samples:
        return jsonify({"error": f"識別名「{name}」は登録されていません"}), 404

    os.makedirs(person_dir(name), exist_ok=True)
    sample_id = new_sample_id()
    save_path = os.path.join(person_dir(name), f"{sample_id}.wav")

    try:
        audio, sr = decode_audio_bytes(file.read())
        sf.write(save_path, audio, sr)
    except Exception as e:
        logging.exception("[register] audio conversion failed")
        return jsonify({"error": f"音声変換失敗: {e}"}), 500

    try:
        t0 = time.perf_counter()
        speech = preprocess(*sf.read(save_path, dtype='float32'))
        speech_sec = speech_seconds(speech)
        if speech_sec < MIN_REGISTER_SEC:
            os.remove(save_path)
            return jsonify({
                "error": f"声の部分が短すぎます（{speech_sec:.1f}秒）。"
                         f"{MIN_REGISTER_SEC:.0f}秒以上話してください",
                "speech_sec": round(speech_sec, 2),
            }), 400
        embedding = embed(speech)
        np.save(cache_path_for(save_path), embedding)
        print(f"[register] get_embedding took {time.perf_counter() - t0:.3f} sec (speech {speech_sec:.1f}s)")
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

    print(f"[register] Total time: {time.perf_counter() - start_time:.3f} sec")
    return jsonify({
        "status": "ok",
        "name": name,
        "kana": metadata[name]["kana"],
        "sample_id": sample_id,
        "num_samples": len(samples[name]),
        "consistency": consistency,
        "speech_sec": round(speech_sec, 2),
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
    start_time = time.perf_counter()
    log("[identify] request received")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    audio_bytes = file.read()

    source = request.form.get("source", "api")
    try:
        t0 = time.perf_counter()
        audio, sr = decode_audio_bytes(audio_bytes)
        t_decode = time.perf_counter()
        speech = preprocess(audio, sr)
        t_pre = time.perf_counter()
        speech_sec = speech_seconds(speech)
        if speech_sec < MIN_IDENTIFY_SEC:
            log(f"[identify] too short: speech {speech_sec:.2f}s < {MIN_IDENTIFY_SEC}s")
            identify_log.append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "result": None,
                "candidate": None,
                "score": None,
                "scores": {},
                "speech_sec": round(speech_sec, 2),
                "reason": "too_short",
            })
            return jsonify({
                "name": None,
                "kana": None,
                "score": None,
                "candidate": None,
                "threshold": SCORE_THRESHOLD,
                "scores": [],
                "speech_sec": round(speech_sec, 2),
                "reason": "too_short",
            })
        if MAX_IDENTIFY_SEC > 0:
            speech = speech[:int(MAX_IDENTIFY_SEC * SAMPLE_RATE)]
        embedding = embed(speech)
        t_embed = time.perf_counter()
        log(f"[identify] decode {(t_decode - t0) * 1000:.0f}ms (sr={sr}), "
            f"preprocess {(t_pre - t_decode) * 1000:.0f}ms, "
            f"embed {(t_embed - t_pre) * 1000:.0f}ms (speech {speech_sec:.1f}s, used {speech_seconds(speech):.1f}s)")
    except Exception as e:
        logging.exception("[identify] get_embedding failed")
        return jsonify({"error": str(e)}), 500

    t0 = time.perf_counter()
    scores = {name: float(np.dot(embedding, prof_embedding))
              for name, prof_embedding in profiles.items()}
    best_name = max(scores, key=scores.get) if scores else None
    best_score = scores[best_name] if best_name else -1
    log(f"[identify] similarity search took {time.perf_counter() - t0:.3f} sec")

    accepted = best_name is not None and best_score >= SCORE_THRESHOLD
    identify_log.append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "result": best_name if accepted else None,
        "candidate": best_name,
        "score": round(best_score, 3),
        "scores": {name: round(score, 3) for name, score in scores.items()},
        "speech_sec": round(speech_sec, 2),
        "reason": None if accepted else "below_threshold",
    })
    score_list = [
        {"name": name, "kana": metadata.get(name, {}).get("kana", name), "score": round(score, 3)}
        for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]

    log(f"[identify] best={best_name} score={best_score:.3f} threshold={SCORE_THRESHOLD}")
    log(f"[identify] Total time: {time.perf_counter() - start_time:.3f} sec")

    if not accepted:
        return jsonify({
            "name": None,
            "kana": None,
            "score": round(best_score, 3),
            "candidate": best_name,
            "threshold": SCORE_THRESHOLD,
            "scores": score_list,
            "speech_sec": round(speech_sec, 2),
            "reason": "below_threshold",
        })

    kana = metadata.get(best_name, {}).get("kana", best_name)
    return jsonify({
        "name": best_name,
        "kana": kana,
        "score": round(best_score, 3),
        "threshold": SCORE_THRESHOLD,
        "scores": score_list,
        "speech_sec": round(speech_sec, 2),
        "reason": None,
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
