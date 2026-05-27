import os
from pathlib import Path

from slimai.helper.features import extract


def test_prefer_local_hf_cache_sets_offline(monkeypatch):
  monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
  monkeypatch.setattr(extract, "_hf_weight_cached", lambda *_args: True)

  with extract._prefer_local_hf_cache("/tmp/cache", "MahmoodLab/UNI", "model.safetensors"):
    assert os.environ.get("HF_HUB_OFFLINE") == "1"

  assert "HF_HUB_OFFLINE" not in os.environ


def test_prefer_local_hf_cache_skips_when_disabled(monkeypatch):
  monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
  monkeypatch.setenv("SLIMAI_PREFER_HF_LOCAL_CACHE", "0")
  monkeypatch.setattr(extract, "_hf_weight_cached", lambda *_args: True)

  with extract._prefer_local_hf_cache("/tmp/cache", "MahmoodLab/UNI", "model.safetensors"):
    assert "HF_HUB_OFFLINE" not in os.environ


def test_prefer_local_hf_cache_respects_existing_offline(monkeypatch):
  monkeypatch.setenv("HF_HUB_OFFLINE", "0")
  monkeypatch.setattr(extract, "_hf_weight_cached", lambda *_args: True)

  with extract._prefer_local_hf_cache("/tmp/cache", "MahmoodLab/UNI", "model.safetensors"):
    assert os.environ.get("HF_HUB_OFFLINE") == "0"


def test_hf_snapshot_weight_path_detects_local_file(tmp_path):
  weight_path = (
    tmp_path
    / "models--MahmoodLab--UNI"
    / "snapshots"
    / "abc123"
    / "model.safetensors"
  )
  weight_path.parent.mkdir(parents=True, exist_ok=True)
  weight_path.write_bytes(b"uni")

  found = extract._hf_snapshot_weight_path(
    str(tmp_path),
    "MahmoodLab/UNI",
    "model.safetensors",
  )
  assert found == weight_path


def test_hf_weight_cached_falls_back_when_try_to_load_returns_object(tmp_path, monkeypatch):
  weight_path = (
    tmp_path
    / "models--MahmoodLab--UNI"
    / "snapshots"
    / "abc123"
    / "model.safetensors"
  )
  weight_path.parent.mkdir(parents=True, exist_ok=True)
  weight_path.write_bytes(b"uni")

  monkeypatch.setattr(
    "huggingface_hub.try_to_load_from_cache",
    lambda **_kwargs: object(),
  )

  assert extract._hf_weight_cached(str(tmp_path), "MahmoodLab/UNI", "model.safetensors")
