"""
speaker-id Flask アプリの起動テスト・疎通テスト

sherpa_onnx はモデルファイルを必要とするため、テスト時はモックに差し替える。
"""
import io
import os
import sys
import json
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
    monkeypatch.setattr(app_module, "profiles", {})
    return tmp_path


def _register(client, name, kana=""):
    return client.post(
        "/register",
        headers=AUTH,
        data={"name": name, "kana": kana, "audio": (_make_wav(), "a.wav", "audio/wav")},
        content_type="multipart/form-data",
    )


def _make_wav(duration_samples: int = 16000) -> io.BytesIO:
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

    def test_embedding_cache_is_used(self, isolated_profiles, monkeypatch):
        """2回目の読み込みでは .npy キャッシュを使い、埋め込みを再計算しない"""
        import soundfile as sf
        (isolated_profiles / "carol").mkdir()
        sf.write(isolated_profiles / "carol" / "20260101-000000-000.wav",
                 np.random.rand(16000).astype(np.float32), 16000)
        app_module.load_profiles()
        assert list((isolated_profiles / "carol").glob(f"*.{app_module.MODEL_TAG}.npy"))

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


# -----------------------------------------------------------------------
# /backup エンドポイントテスト
# -----------------------------------------------------------------------
class TestBackup:
    def test_backup_returns_zip(self, client):
        """認証ありで ZIP ファイルが返る"""
        resp = client.get("/backup", headers=AUTH)
        assert resp.status_code == 200
        assert resp.content_type == "application/zip"
