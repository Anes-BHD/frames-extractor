"""CPU-only tests for the cloud-ready changes (no GPU, no model downloads)."""
import pytest
from fastapi.testclient import TestClient

from backend import stage3_rank


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FRAMES_DEVICE", "FRAMES_SIGLIP_CHECKPOINT", "FRAMES_BATCH_SIZE"):
        monkeypatch.delenv(var, raising=False)


def _fake_cuda(monkeypatch, available: bool) -> None:
    monkeypatch.setattr(stage3_rank.torch.cuda, "is_available", lambda: available)


def test_auto_uses_cuda_when_available(monkeypatch):
    _fake_cuda(monkeypatch, True)
    assert stage3_rank._select_device() == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch):
    _fake_cuda(monkeypatch, False)
    assert stage3_rank._select_device() == "cpu"


def test_forced_cuda_without_gpu_fails_loudly(monkeypatch):
    _fake_cuda(monkeypatch, False)
    monkeypatch.setenv("FRAMES_DEVICE", "cuda")
    with pytest.raises(RuntimeError):
        stage3_rank._select_device()


def test_forced_cpu_even_with_gpu(monkeypatch):
    _fake_cuda(monkeypatch, True)
    monkeypatch.setenv("FRAMES_DEVICE", "cpu")
    assert stage3_rank._select_device() == "cpu"


def test_invalid_device_value(monkeypatch):
    monkeypatch.setenv("FRAMES_DEVICE", "tpu")
    with pytest.raises(ValueError):
        stage3_rank._select_device()


def test_stage3_config_defaults_unchanged():
    cfg = stage3_rank.Stage3Config()
    assert cfg.checkpoint_id == "google/siglip2-so400m-patch14-384"
    assert cfg.batch_size == 64
    assert cfg.top_k == 50


def test_stage3_config_env_overrides(monkeypatch):
    monkeypatch.setenv("FRAMES_BATCH_SIZE", "8")
    monkeypatch.setenv("FRAMES_SIGLIP_CHECKPOINT", "some/other-checkpoint")
    cfg = stage3_rank.Stage3Config()
    assert cfg.batch_size == 8
    assert cfg.checkpoint_id == "some/other-checkpoint"


def test_stage3_config_still_rejects_unsafe_batch(monkeypatch):
    monkeypatch.setenv("FRAMES_BATCH_SIZE", "256")
    with pytest.raises(ValueError):
        stage3_rank.Stage3Config()


def test_health_endpoint(tmp_path, monkeypatch):
    import frontend.app as app_module
    from frontend import storage

    monkeypatch.setattr(storage, "ASSETS_ROOT", tmp_path / "assets")
    client = TestClient(app_module.app)  # no `with`: skips the startup scan
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["storage_writable"] is True
