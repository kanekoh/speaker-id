"""
speaker-id Flask アプリの起動テスト・疎通テスト

sherpa_onnx はモデルファイルを必要とするため、テスト時はモックに差し替える。
"""
import io
import os
import sys
import json
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock

# -----------------------------------------------------------------------
# sherpa_onnx をモック化してからアプリをインポートする
# (モデルファイルなしでもテストが実行できるようにする)
# -----------------------------------------------------------------------
_mock_stream = MagicMock()
_mock_extractor = MagicMock()
_mock_extractor.create_stream.return_value = _mock_stream
_mock_extractor.compute.return_value = np.ones(192, dtype=np.float32).tolist()
_mock_sherpa = MagicMock()
_mock_sherpa.SpeakerEmbeddingExtractor.return_value = _mock_extractor
_mock_sherpa.SpeakerEmbeddingExtractorConfig.return_value = MagicMock()
sys.modules.setdefault("sherpa_onnx", _mock_sherpa)

# soundfile.read もモック化する
# (profiles/ に不正な WAV ファイルがあってもインポート時に失敗しないようにする)
import soundfile as _sf_real  # noqa: E402
_orig_sf_read = _sf_real.read

def _mock_sf_read(file, **kwargs):
    """テスト用: 任意のファイルに対しダミー音声データを返す"""
    dummy_audio = np.ones(16000, dtype=np.float32)
    return dummy_audio, 16000

_sf_real.read = _mock_sf_read

# 必須環境変数を設定してからインポート
os.environ.setdefault("SPEAKERID_API_KEY", "test-secret-token-for-pytest")
os.environ.setdefault("SPEAKER_MODEL_PATH", "/tmp/dummy.onnx")
# 本物の profiles/ を移行・変更しないよう、テスト用の一時ディレクトリを使う
os.environ["SPEAKER_PROFILE_DIR"] = tempfile.mkdtemp(prefix="speaker-id-test-")
os.environ["SPEAKER_LOG_DIR"] = tempfile.mkdtemp(prefix="speaker-id-log-")

import app as app_module  # noqa: E402

# アプリインポート後は実際の soundfile.read を使用する
# (テスト内で _make_wav() が soundfile.write したバッファを読み込めるようにする)
_sf_real.read = _orig_sf_read

TEST_TOKEN = "test-secret-token-for-pytest"
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}
WRONG_AUTH = {"Authorization": "Bearer wrong-token"}


# -----------------------------------------------------------------------
# fixture
# -----------------------------------------------------------------------
@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """空の profiles ディレクトリで登録系をテストする"""
    monkeypatch.setattr(app_module, "PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "META_FILE", str(tmp_path / "metadata.json"))
    monkeypatch.setattr(app_module, "metadata", {})
    monkeypatch.setattr(app_module, "samples", {})
    monkeypatch.setattr(app_module, "sample_envs", {})
    monkeypatch.setattr(app_module, "profiles", {})
    return tmp_path


def _register(client, name, kana="", mode=None):
    data = {"name": name, "kana": kana, "audio": (_make_wav(), "a.wav", "audio/wav")}
    if mode:
        data["mode"] = mode
    return client.post("/register", headers=AUTH, data=data,
                       content_type="multipart/form-data")


def _make_speech_wav(speech_sec: float, silence_sec: float) -> io.BytesIO:
    """前後に無音を挟んだ「声」（正弦波）の WAV を作る"""
    import soundfile as sf

    sr = 16000
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(int(speech_sec * sr)) / sr)
    silence = np.zeros(int(silence_sec * sr / 2))
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([silence, tone, silence]).astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf


def _make_wav(duration_samples: int = 16000 * 4) -> io.BytesIO:
    """soundfile でダミー WAV を生成して BytesIO で返す"""
    import soundfile as sf

    audio = np.random.rand(duration_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------
# 起動テスト
# -----------------------------------------------------------------------
class TestStartup:
    def test_app_object_exists(self):
        """Flask アプリオブジェクトが生成されている"""
        assert app_module.app is not None

    def test_api_token_loaded(self):
        """SPEAKERID_API_KEY が正しく読み込まれている"""
        assert app_module.API_TOKEN == TEST_TOKEN

    def test_extractor_initialized(self):
        """sherpa_onnx エクストラクタが初期化されている"""
        assert app_module.extractor is not None

    def test_profile_dir_exists(self):
        """profiles ディレクトリが作成されている"""
        assert os.path.isdir(app_module.PROFILE_DIR)


# -----------------------------------------------------------------------
# 疎通テスト: 認証なし
# -----------------------------------------------------------------------
class TestAuthRequired:
    def test_index_accessible_without_auth(self, client):
        """GET / は認証不要"""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_check_token_requires_auth(self, client):
        """GET /check_token は Bearer トークンなしで 401"""
        resp = client.get("/check_token")
        assert resp.status_code == 401

    def test_register_requires_auth(self, client):
        """POST /register は Bearer トークンなしで 401"""
        resp = client.post("/register")
        assert resp.status_code == 401

    def test_identify_requires_auth(self, client):
        """POST /identify は Bearer トークンなしで 401"""
        resp = client.post("/identify")
        assert resp.status_code == 401

    def test_backup_requires_auth(self, client):
        """GET /backup は Bearer トークンなしで 401"""
        resp = client.get("/backup")
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, client):
        """誤ったトークンは 401"""
        resp = client.get("/check_token", headers=WRONG_AUTH)
        assert resp.status_code == 401


# -----------------------------------------------------------------------
# 疎通テスト: /check_token
# -----------------------------------------------------------------------
class TestCheckToken:
    def test_valid_token_returns_ok(self, client):
        """正しいトークンで {ok: true} が返る"""
        resp = client.get("/check_token", headers=AUTH)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"ok": True}

    def test_response_is_json(self, client):
        """レスポンスが JSON である"""
        resp = client.get("/check_token", headers=AUTH)
        assert resp.content_type.startswith("application/json")


# -----------------------------------------------------------------------
# /register エンドポイントテスト
# -----------------------------------------------------------------------
class TestRegister:
    def test_missing_audio_returns_400(self, client):
        """audio フィールドなしで 400"""
        resp = client.post(
            "/register",
            headers=AUTH,
            data={"name": "alice"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_missing_name_returns_400(self, client):
        """name フィールドなしで 400"""
        buf = _make_wav()
        resp = client.post(
            "/register",
            headers=AUTH,
            data={"audio": (buf, "test.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_empty_name_returns_400(self, client):
        """空の name で 400"""
        buf = _make_wav()
        resp = client.post(
            "/register",
            headers=AUTH,
            data={"name": "   ", "audio": (buf, "test.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_register_success(self, client, isolated_profiles):
        """正常な WAV と name で登録成功"""
        buf = _make_wav()
        resp = client.post(
            "/register",
            headers=AUTH,
            data={
                "name": "alice",
                "kana": "アリス",
                "audio": (buf, "alice.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["name"] == "alice"
        assert data["kana"] == "アリス"
        assert data["num_samples"] == 1
        assert data["consistency"] is None

    def test_register_appends_samples(self, client, isolated_profiles):
        """同じ名前で登録するとサンプルが追加され、一致度が返る"""
        _register(client, "alice", "アリス")
        data = _register(client, "alice").get_json()
        assert data["num_samples"] == 2
        assert data["consistency"] == pytest.approx(1.0)
        assert data["kana"] == "アリス"  # kana 省略時は既存の表示名を保つ
        assert len(list((isolated_profiles / "alice").glob("*.wav"))) == 2
        assert len(app_module.samples["alice"]) == 2
        assert np.isclose(np.linalg.norm(app_module.profiles["alice"]), 1.0)

    def test_register_rejects_short_speech(self, client, isolated_profiles):
        """無音を除いた声が短すぎる録音は 400 で、ファイルも残らない"""
        resp = client.post(
            "/register",
            headers=AUTH,
            data={"name": "alice", "audio": (_make_speech_wav(1.0, silence_sec=3.0), "a.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert resp.get_json()["speech_sec"] < app_module.MIN_REGISTER_SEC
        assert "alice" not in app_module.profiles
        assert not list(isolated_profiles.glob("alice/*.wav"))

    def test_register_mode_new_rejects_existing_name(self, client, isolated_profiles):
        """mode=new で既存の識別名を指定すると 409 で、録音は増えない"""
        _register(client, "alice")
        resp = _register(client, "alice", mode="new")
        assert resp.status_code == 409
        assert len(app_module.samples["alice"]) == 1
        assert len(list((isolated_profiles / "alice").glob("*.wav"))) == 1

    def test_register_mode_add_rejects_unknown_name(self, client, isolated_profiles):
        """mode=add で未登録の識別名を指定すると 404 で、新しい人は作られない"""
        _register(client, "alice")
        resp = _register(client, "alicee", mode="add")
        assert resp.status_code == 404
        assert "alicee" not in app_module.samples
        assert not (isolated_profiles / "alicee").exists()

    def test_register_modes_success(self, client, isolated_profiles):
        """mode=new で新規登録、mode=add で追加できる"""
        assert _register(client, "alice", mode="new").status_code == 200
        data = _register(client, "alice", mode="add").get_json()
        assert data["num_samples"] == 2

    @pytest.mark.parametrize("name", ["../evil", "a/b", "田中", "a b"])
    def test_register_rejects_invalid_name(self, client, isolated_profiles, name):
        """パスに使えない識別名は 400"""
        assert _register(client, name).status_code == 400


# -----------------------------------------------------------------------
# /profiles エンドポイントテスト
# -----------------------------------------------------------------------
class TestProfiles:
    def test_requires_auth(self, client):
        assert client.get("/profiles").status_code == 401
        assert client.delete("/profiles/alice").status_code == 401

    def test_list_profiles(self, client, isolated_profiles):
        """登録者ごとにサンプル一覧と一致度が返る"""
        _register(client, "alice", "アリス")
        _register(client, "alice")
        people = client.get("/profiles", headers=AUTH).get_json()["people"]
        assert [p["name"] for p in people] == ["alice"]
        assert people[0]["kana"] == "アリス"
        assert len(people[0]["samples"]) == 2
        assert people[0]["samples"][0]["consistency"] == pytest.approx(1.0)

    def test_delete_sample(self, client, isolated_profiles):
        """サンプルを1つ削除すると残りでプロファイルが作り直される"""
        first = _register(client, "alice").get_json()["sample_id"]
        _register(client, "alice")
        resp = client.delete(f"/profiles/alice/samples/{first}", headers=AUTH)
        assert resp.status_code == 200
        assert resp.get_json()["num_samples"] == 1
        assert first not in app_module.samples["alice"]
        assert not list((isolated_profiles / "alice").glob(f"{first}*"))

    def test_delete_last_sample_removes_person(self, client, isolated_profiles):
        sample_id = _register(client, "alice").get_json()["sample_id"]
        client.delete(f"/profiles/alice/samples/{sample_id}", headers=AUTH)
        assert "alice" not in app_module.profiles
        assert "alice" not in app_module.metadata
        assert not (isolated_profiles / "alice").exists()

    def test_delete_profile(self, client, isolated_profiles):
        _register(client, "alice")
        assert client.delete("/profiles/alice", headers=AUTH).status_code == 200
        assert "alice" not in app_module.profiles
        assert not (isolated_profiles / "alice").exists()
        assert client.delete("/profiles/alice", headers=AUTH).status_code == 404

    def test_delete_sample_rejects_bad_id(self, client, isolated_profiles):
        _register(client, "alice")
        resp = client.delete("/profiles/alice/samples/..%2Fmetadata", headers=AUTH)
        assert resp.status_code == 404


# -----------------------------------------------------------------------
# 録音環境ごとのプロファイル・ログの録音からの登録
# -----------------------------------------------------------------------
def _identify_entry(client, source="api", wav=None):
    """/identify を呼び、履歴に追加されたエントリを返す"""
    client.post("/identify", headers=AUTH,
                data={"audio": (wav or _make_wav(), "q.wav", "audio/wav"), "source": source},
                content_type="multipart/form-data")
    return app_module.identify_log[-1]


class TestEnvironments:
    def test_browser_register_records_env(self, client, isolated_profiles):
        """登録画面からの登録は browser として記録され、一覧にも出る"""
        sample_id = _register(client, "alice").get_json()["sample_id"]
        assert json.loads((isolated_profiles / "alice" / f"{sample_id}.json").read_text()) == {"env": "browser"}
        people = client.get("/profiles", headers=AUTH).get_json()["people"]
        assert people[0]["samples"][0]["env"] == "browser"

    def test_register_from_identify_log(self, client, isolated_profiles):
        """端末からの識別の録音を登録すると device として加わり、環境ごとの平均ができる"""
        _register(client, "alice")
        entry = _identify_entry(client, source="stackchan")
        resp = client.post(f"/identify_log/{entry['id']}/register", headers=AUTH, json={"name": "alice"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["env"] == "device"
        assert data["num_samples"] == 2
        assert app_module.profiles["alice"].shape[0] == 2  # browser と device の 2 つ
        log = client.get("/identify_log", headers=AUTH).get_json()["entries"]
        assert [e["registered_to"] for e in log if e["id"] == entry["id"]] == ["alice"]

    def test_register_from_identify_log_twice_is_rejected(self, client, isolated_profiles):
        _register(client, "alice")
        entry = _identify_entry(client)
        client.post(f"/identify_log/{entry['id']}/register", headers=AUTH, json={"name": "alice"})
        resp = client.post(f"/identify_log/{entry['id']}/register", headers=AUTH, json={"name": "alice"})
        assert resp.status_code == 409
        assert len(app_module.samples["alice"]) == 2

    def test_register_from_identify_log_validation(self, client, isolated_profiles):
        _register(client, "alice")
        entry = _identify_entry(client)
        url = f"/identify_log/{entry['id']}/register"
        assert client.post(url, json={"name": "alice"}).status_code == 401
        # 既定は「登録済みの人に追加」なので、未登録の名前は 404
        assert client.post(url, headers=AUTH, json={"name": "bob"}).status_code == 404
        assert client.post("/identify_log/999999/register", headers=AUTH,
                           json={"name": "alice"}).status_code == 404
        # mode=new なら新しい人として登録できる
        resp = client.post(url, headers=AUTH, json={"name": "bob", "kana": "ボブ", "mode": "new"})
        assert resp.status_code == 200 and resp.get_json()["kana"] == "ボブ"

    def test_short_identify_audio_cannot_be_registered(self, client, isolated_profiles):
        """登録の最低秒数に満たない録音は 400 で、登録済みにもならない"""
        _register(client, "alice")
        entry = _identify_entry(client, wav=_make_speech_wav(1.5, silence_sec=1.0))
        resp = client.post(f"/identify_log/{entry['id']}/register", headers=AUTH, json={"name": "alice"})
        assert resp.status_code == 400
        assert app_module.identify_audio[entry["id"]].get("registered_to") is None

    def test_profile_score_uses_closest_environment(self):
        """プロファイルは環境ごとの平均の組で、最も近いものとのスコアを使う"""
        a = np.zeros(4, dtype=np.float32); a[0] = 1
        b = np.zeros(4, dtype=np.float32); b[1] = 1
        assert app_module.profile_score(b, np.stack([a, b])) == pytest.approx(1.0)
        assert app_module.profile_score(b, a) == pytest.approx(0.0)  # 1 次元（環境 1 つ）でも可

    def test_consistency_compares_within_same_env(self, client, isolated_profiles):
        """一覧の一致度は同じ環境の録音どうしで比べる（環境が 1 本ずつなら比較対象なし）"""
        _register(client, "alice")
        entry = _identify_entry(client)
        client.post(f"/identify_log/{entry['id']}/register", headers=AUTH, json={"name": "alice"})
        samples = client.get("/profiles", headers=AUTH).get_json()["people"][0]["samples"]
        assert sorted(s["env"] for s in samples) == ["browser", "device"]
        assert all(s["consistency"] is None for s in samples)

    def test_env_is_restored_on_load_and_removed_with_sample(self, client, isolated_profiles):
        _register(client, "alice")
        entry = _identify_entry(client)
        sample_id = client.post(f"/identify_log/{entry['id']}/register", headers=AUTH,
                                json={"name": "alice"}).get_json()["sample_id"]
        app_module.load_profiles()
        assert app_module.sample_envs["alice"][sample_id] == "device"
        client.delete(f"/profiles/alice/samples/{sample_id}", headers=AUTH)
        assert not (isolated_profiles / "alice" / f"{sample_id}.json").exists()
        assert app_module.profiles["alice"].shape[0] == 1


# -----------------------------------------------------------------------
# 起動時の読み込み・旧形式からの移行
# -----------------------------------------------------------------------
class TestLoadProfiles:
    def test_migrates_legacy_wav(self, isolated_profiles):
        """profiles/<name>.wav は profiles/<name>/<id>.wav に移行される"""
        import soundfile as sf
        sf.write(isolated_profiles / "bob.wav", np.random.rand(16000).astype(np.float32), 16000)
        app_module.load_profiles()
        assert not (isolated_profiles / "bob.wav").exists()
        assert len(list((isolated_profiles / "bob").glob("*.wav"))) == 1
        assert "bob" in app_module.profiles

    def test_stale_cache_is_removed(self, isolated_profiles):
        """モデル・前処理が違う古いキャッシュは削除され、作り直される"""
        import soundfile as sf
        (isolated_profiles / "dave").mkdir()
        sf.write(isolated_profiles / "dave" / "20260101-000000-000.wav",
                 np.random.rand(16000).astype(np.float32), 16000)
        np.save(isolated_profiles / "dave" / "20260101-000000-000.oldtag.npy", np.zeros(192))
        app_module.load_profiles()
        npys = sorted(p.name for p in (isolated_profiles / "dave").glob("*.npy"))
        assert npys == [f"20260101-000000-000.{app_module.EMBED_TAG}.npy"]

    def test_embedding_cache_is_used(self, isolated_profiles, monkeypatch):
        """2回目の読み込みでは .npy キャッシュを使い、埋め込みを再計算しない"""
        import soundfile as sf
        (isolated_profiles / "carol").mkdir()
        sf.write(isolated_profiles / "carol" / "20260101-000000-000.wav",
                 np.random.rand(16000).astype(np.float32), 16000)
        app_module.load_profiles()
        assert list((isolated_profiles / "carol").glob(f"*.{app_module.EMBED_TAG}.npy"))

        def fail(*args, **kwargs):
            raise AssertionError("cache not used")
        monkeypatch.setattr(app_module, "get_embedding_from_file", fail)
        app_module.load_profiles()
        assert "carol" in app_module.profiles


# -----------------------------------------------------------------------
# /identify エンドポイントテスト
# -----------------------------------------------------------------------
class TestIdentify:
    def test_no_audio_returns_400(self, client):
        """audio なしで 400"""
        resp = client.post("/identify", headers=AUTH)
        assert resp.status_code == 400

    def test_identify_with_no_profiles(self, client, monkeypatch):
        """プロファイルが 0 件のとき name=null が返る"""
        monkeypatch.setattr(app_module, "profiles", {})
        monkeypatch.setattr(app_module, "metadata", {})

        buf = _make_wav()
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (buf, "query.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] is None

    def test_identify_returns_best_match(self, client, monkeypatch):
        """プロファイルがある場合、最も類似度の高い話者が返る"""
        vec_alice = np.ones(192, dtype=np.float32)
        vec_bob = -np.ones(192, dtype=np.float32)
        monkeypatch.setattr(app_module, "profiles", {"alice": vec_alice, "bob": vec_bob})
        monkeypatch.setattr(app_module, "metadata", {"alice": {"kana": "アリス"}})

        # モックは常に ones ベクトルを返すので alice に一致するはず
        buf = _make_wav()
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (buf, "query.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "alice"
        assert data["kana"] == "アリス"
        assert "score" in data

    def test_identify_below_threshold_returns_none(self, client, monkeypatch):
        """最高スコアが閾値未満なら name=null と候補者が返る"""
        # モックの埋め込み(ones)と直交するベクトル → コサイン類似度 0
        vec = np.zeros(192, dtype=np.float32)
        vec[0], vec[1] = 1.0, -1.0
        vec /= np.linalg.norm(vec)
        monkeypatch.setattr(app_module, "profiles", {"dave": vec})
        monkeypatch.setattr(app_module, "metadata", {"dave": {"kana": "デイブ"}})

        buf = _make_wav()
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (buf, "query.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] is None
        assert data["kana"] is None
        assert data["candidate"] == "dave"

    def test_identify_too_short(self, client, monkeypatch):
        """声が短すぎると name=null・reason=too_short が返り、履歴にも残る"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        app_module.identify_log.clear()
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (_make_speech_wav(0.3, silence_sec=2.0), "q.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] is None
        assert data["reason"] == "too_short"
        assert data["speech_sec"] < app_module.MIN_IDENTIFY_SEC
        assert app_module.identify_log[-1]["reason"] == "too_short"

    def test_identify_silence_only(self, client, monkeypatch):
        """無音だけの音声も too_short になる"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (_make_speech_wav(0.0, silence_sec=3.0), "q.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        assert resp.get_json()["reason"] == "too_short"

    def test_identify_response_schema(self, client, monkeypatch):
        """レスポンスに name / kana / score キーが含まれる"""
        monkeypatch.setattr(app_module, "profiles", {"carol": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "metadata", {})

        buf = _make_wav()
        resp = client.post(
            "/identify",
            headers=AUTH,
            data={"audio": (buf, "query.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        assert set(data.keys()) >= {"name", "kana", "score"}


# -----------------------------------------------------------------------
# /identify_log エンドポイントテスト
# -----------------------------------------------------------------------
class TestIdentifyLog:
    def _identify(self, client, source=None):
        data = {"audio": (_make_wav(), "query.wav", "audio/wav")}
        if source:
            data["source"] = source
        return client.post("/identify", headers=AUTH, data=data,
                           content_type="multipart/form-data")

    def test_requires_auth(self, client):
        """GET/DELETE /identify_log は Bearer トークンなしで 401"""
        assert client.get("/identify_log").status_code == 401
        assert client.delete("/identify_log").status_code == 401

    def test_identify_returns_all_scores_sorted(self, client, monkeypatch):
        """/identify は全登録者のスコアを降順で返す"""
        vec_bob = np.zeros(192, dtype=np.float32)
        vec_bob[0] = 1.0
        monkeypatch.setattr(app_module, "profiles", {
            "alice": np.ones(192, dtype=np.float32) / np.sqrt(192), "bob": vec_bob})
        monkeypatch.setattr(app_module, "metadata", {"alice": {"kana": "アリス"}})

        data = self._identify(client).get_json()
        assert [s["name"] for s in data["scores"]] == ["alice", "bob"]
        assert data["scores"][0]["kana"] == "アリス"
        assert data["scores"][1]["kana"] == "bob"
        assert data["threshold"] == app_module.SCORE_THRESHOLD

    def test_identify_is_logged(self, client, monkeypatch):
        """識別結果が履歴に記録され、GET で取得できる"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32) / np.sqrt(192)})
        monkeypatch.setattr(app_module, "metadata", {"alice": {"kana": "アリス"}})
        app_module.identify_log.clear()

        self._identify(client, source="test")
        resp = client.get("/identify_log", headers=AUTH)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["people"] == [{"name": "alice", "kana": "アリス"}]
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["result"] == "alice"
        assert entry["source"] == "test"
        assert set(entry["scores"]) == {"alice"}

    def test_clear_log(self, client, monkeypatch):
        """DELETE /identify_log で履歴が空になる"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        self._identify(client)
        assert client.delete("/identify_log", headers=AUTH).status_code == 200
        assert client.get("/identify_log", headers=AUTH).get_json()["entries"] == []

    def test_entry_has_id_levels_and_audio(self, client, monkeypatch):
        """履歴には id・SN 比が入り、録音を再生できる"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        client.delete("/identify_log", headers=AUTH)
        self._identify(client)
        entry = client.get("/identify_log", headers=AUTH).get_json()["entries"][-1]
        assert isinstance(entry["id"], int)
        assert entry["has_audio"] is True
        assert {"snr_db", "noise_db", "speech_db"} <= set(entry)

        for kind in ["raw", "processed"]:
            resp = client.get(f"/identify_log/{entry['id']}/audio?kind={kind}", headers=AUTH)
            assert resp.status_code == 200
            assert resp.mimetype == "audio/wav"
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(resp.data))
            assert len(audio) > 0
        assert client.get(f"/identify_log/{entry['id']}/audio?kind=xx", headers=AUTH).status_code == 400

    def test_audio_requires_auth_and_existing_entry(self, client):
        assert client.get("/identify_log/1/audio").status_code == 401
        assert client.get("/identify_log/999999/audio", headers=AUTH).status_code == 404

    def test_only_recent_audio_is_kept(self, client, monkeypatch):
        """録音は直近 IDENTIFY_AUDIO_SIZE 件だけ保持する（履歴は残る）"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "IDENTIFY_AUDIO_SIZE", 2)
        client.delete("/identify_log", headers=AUTH)
        for _ in range(3):
            self._identify(client)
        entries = client.get("/identify_log", headers=AUTH).get_json()["entries"]
        assert [e["has_audio"] for e in entries] == [False, True, True]

    def test_clear_log_removes_audio(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        self._identify(client)
        client.delete("/identify_log", headers=AUTH)
        assert len(app_module.identify_audio) == 0

    def test_log_survives_restart(self, client, isolated_profiles):
        """履歴・録音・登録済みの状態はディスクに残り、再起動（再読み込み）後も使える"""
        client.delete("/identify_log", headers=AUTH)
        _register(client, "alice")
        self._identify(client, source="stackchan")
        self._identify(client, source="stackchan")
        first, second = [e["id"] for e in app_module.identify_log]
        client.post(f"/identify_log/{first}/register", headers=AUTH, json={"name": "alice"})

        app_module.load_identify_log()  # 再起動の代わり

        entries = client.get("/identify_log", headers=AUTH).get_json()["entries"]
        assert [e["id"] for e in entries] == [first, second]
        assert [e["registered_to"] for e in entries] == ["alice", None]
        assert client.get(f"/identify_log/{second}/audio", headers=AUTH).status_code == 200
        # ID は続きから振られる
        self._identify(client)
        assert app_module.identify_log[-1]["id"] == second + 1

    def test_old_audio_files_are_deleted(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "IDENTIFY_AUDIO_SIZE", 2)
        client.delete("/identify_log", headers=AUTH)
        for _ in range(3):
            self._identify(client)
        assert len(os.listdir(app_module.LOG_AUDIO_DIR)) == 2

    def test_load_skips_broken_lines(self, client):
        client.delete("/identify_log", headers=AUTH)
        with open(app_module.LOG_FILE, "a") as f:
            f.write('{"type": "entry", "id": 5, "time": "2026-01-01T00:00:00", "source": "api"}\n{"broken\n')
        app_module.load_identify_log()
        assert [e["id"] for e in app_module.identify_log] == [5]

    def test_denoise_compare_and_audio(self, client, monkeypatch):
        """ノイズ除去した場合のスコアと、除去後の音を返す（識別には使わない）"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "denoise", lambda x: x * 0.5)
        client.delete("/identify_log", headers=AUTH)
        self._identify(client)
        entry_id = app_module.identify_log[-1]["id"]
        resp = client.get(f"/identify_log/{entry_id}/denoise_compare", headers=AUTH)
        assert resp.status_code == 200
        data = resp.get_json()
        assert set(data["before"]["scores"]) == {"alice"}
        assert set(data["after"]["scores"]) == {"alice"}
        assert "snr_db" in data["after"]
        resp = client.get(f"/identify_log/{entry_id}/audio?kind=denoised", headers=AUTH)
        assert resp.status_code == 200 and resp.mimetype == "audio/wav"

    def test_denoise_without_model_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "DENOISE_MODEL_PATH", "/nonexistent.onnx")
        monkeypatch.setattr(app_module, "_denoiser", None)
        self._identify(client)
        entry_id = app_module.identify_log[-1]["id"]
        assert client.get(f"/identify_log/{entry_id}/denoise_compare", headers=AUTH).status_code == 503

    def test_scores_page_accessible(self, client):
        """GET /scores.html は認証不要でページが返る"""
        assert client.get("/scores.html").status_code == 200


# -----------------------------------------------------------------------
# 埋め込み抽出テスト
# -----------------------------------------------------------------------
class TestEmbedding:
    def test_embedding_is_l2_normalized(self):
        """埋め込みは L2 ノルム 1 に正規化されている"""
        audio = np.random.rand(16000).astype(np.float32)
        emb = app_module.get_embedding_from_array(audio, 16000)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


class TestDecodeAudio:
    def test_wav_is_read_without_ffmpeg(self, monkeypatch):
        """WAV は ffmpeg を起動せずに直接読む"""
        def fail(*args, **kwargs):
            raise AssertionError("ffmpeg should not be called for WAV")
        monkeypatch.setattr(app_module.subprocess, "run", fail)
        audio, sr = app_module.decode_audio_bytes(_make_wav(48000).getvalue())
        assert sr == 16000 and len(audio) == 48000

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_webm_is_decoded_by_ffmpeg(self):
        """WebM/Opus（Chrome の録音形式）は ffmpeg で 16kHz モノラルにする"""
        webm = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2:sample_rate=48000",
             "-c:a", "libopus", "-f", "webm", "pipe:1"],
            capture_output=True, check=True).stdout
        audio, sr = app_module.decode_audio_bytes(webm)
        assert sr == app_module.SAMPLE_RATE
        assert audio.ndim == 1
        assert abs(len(audio) / sr - 2.0) < 0.1

    def test_invalid_bytes_raise(self):
        with pytest.raises(ValueError):
            app_module.decode_audio_bytes(b"not audio at all")

    @pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
    def test_preprocess_resamples_to_16k(self, sr):
        """どのサンプルレートでも 16kHz の長さになる"""
        tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(2 * sr) / sr).astype(np.float32)
        speech = app_module.preprocess(tone, sr)
        assert abs(len(speech) - 2 * app_module.SAMPLE_RATE) <= app_module.SAMPLE_RATE * 0.05

    def test_identify_uses_at_most_max_seconds(self, client, monkeypatch):
        """長い音声は MAX_IDENTIFY_SEC で打ち切ってから埋め込みを計算する"""
        monkeypatch.setattr(app_module, "profiles", {"alice": np.ones(192, dtype=np.float32)})
        monkeypatch.setattr(app_module, "MAX_IDENTIFY_SEC", 2.0)
        lengths = []
        orig_embed = app_module.embed
        monkeypatch.setattr(app_module, "embed", lambda x: lengths.append(len(x)) or orig_embed(x))
        client.post("/identify", headers=AUTH,
                    data={"audio": (_make_wav(16000 * 5), "q.wav", "audio/wav")},
                    content_type="multipart/form-data")
        assert lengths == [2 * app_module.SAMPLE_RATE]

    def test_identify_invalid_audio_returns_500(self, client):
        resp = client.post("/identify", headers=AUTH,
                           data={"audio": (io.BytesIO(b"garbage"), "q.wav", "audio/wav")},
                           content_type="multipart/form-data")
        assert resp.status_code == 500


class TestMeasureLevels:
    def test_snr_of_tone_with_noise(self):
        """声（正弦波）とノイズの差がおおよそ SN 比として出る"""
        sr = app_module.SAMPLE_RATE
        rng = np.random.default_rng(0)
        noise = 0.003 * rng.standard_normal(3 * sr)       # 約 -50dB
        tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)  # 約 -13dB
        audio = noise.copy()
        audio[sr:2 * sr] += tone
        levels = app_module.measure_levels(audio.astype(np.float32))
        assert -53 < levels["noise_db"] < -47
        assert 30 < levels["snr_db"] < 42

    def test_too_short(self):
        assert app_module.measure_levels(np.zeros(10, dtype=np.float32))["snr_db"] is None


class TestTrimSilence:
    def test_removes_leading_and_trailing_silence(self):
        """前後の無音が除かれ、声の長さ＋余白程度になる"""
        sr = app_module.SAMPLE_RATE
        tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(2 * sr) / sr)
        audio = np.concatenate([np.zeros(sr), tone, np.zeros(sr)]).astype(np.float32)
        trimmed = app_module.trim_silence(audio)
        pad = 2 * app_module.SPEECH_PAD_FRAMES * app_module.FRAME_SEC
        assert 2.0 <= len(trimmed) / sr <= 2.0 + pad + 0.05

    def test_quiet_noise_is_silence(self):
        """-50dBFS 未満の小さな雑音だけなら何も残らない"""
        audio = (0.001 * np.random.randn(16000 * 2)).astype(np.float32)
        assert len(app_module.trim_silence(audio)) == 0

    def test_too_short_input(self):
        assert len(app_module.trim_silence(np.zeros(10, dtype=np.float32))) == 0


# -----------------------------------------------------------------------
# /backup エンドポイントテスト
# -----------------------------------------------------------------------
class TestBackup:
    def test_backup_returns_zip(self, client):
        """認証ありで ZIP ファイルが返る"""
        resp = client.get("/backup", headers=AUTH)
        assert resp.status_code == 200
        assert resp.content_type == "application/zip"
