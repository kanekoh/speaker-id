"""
speaker-id Flask アプリの起動テスト・疎通テスト

sherpa_onnx はモデルファイルを必要とするため、テスト時はモックに差し替える。
"""
import io
import os
import sys
import json

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
_mock_extractor.get.return_value = np.ones(192, dtype=np.float32).tolist()
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

    def test_register_success(self, client, tmp_path, monkeypatch):
        """正常な WAV と name で登録成功"""
        monkeypatch.setattr(app_module, "PROFILE_DIR", str(tmp_path))
        monkeypatch.setattr(app_module, "META_FILE", str(tmp_path / "metadata.json"))
        monkeypatch.setattr(app_module, "metadata", {})
        monkeypatch.setattr(app_module, "profiles", {})

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
# /backup エンドポイントテスト
# -----------------------------------------------------------------------
class TestBackup:
    def test_backup_returns_zip(self, client):
        """認証ありで ZIP ファイルが返る"""
        resp = client.get("/backup", headers=AUTH)
        assert resp.status_code == 200
        assert resp.content_type == "application/zip"
