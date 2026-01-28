from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from ml_object_detector import DetectionResult, load, load_image, predict
import ml_object_detector.detectors as detectors


class DummyYOLO:
    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path


class DummyTensor:
    def __init__(self, data: list[float] | list[list[float]]) -> None:
        self._data = data

    def cpu(self) -> "DummyTensor":
        return self

    def tolist(self) -> list[float] | list[list[float]]:
        return self._data


class DummyBoxes:
    def __init__(self) -> None:
        self.xyxy = DummyTensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self.conf = DummyTensor([0.9, 0.8])
        self.cls = DummyTensor([0.0, 1.0])


class DummyResult:
    def __init__(self) -> None:
        self.boxes = DummyBoxes()


class DummyModel:
    def __init__(self) -> None:
        self.names = {0: "chip", 1: "short"}
        self.last_conf: float | None = None

    def __call__(self, image_path: str, conf: float) -> list[DummyResult]:
        self.last_conf = conf
        return [DummyResult()]


def test_load_image_returns_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGBA", (8, 8), color=(255, 0, 0, 128)).save(image_path)

    image = load_image(image_path)

    assert image.mode == "RGB"


def test_load_downloads_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights_dir = tmp_path / "artifacts"
    weights_dir.mkdir()
    weights_path = weights_dir / "model.pt"
    weights_path.write_text("weights")

    def fake_set_tracking_uri(uri: str) -> None:
        assert uri == "databricks"

    def fake_download_artifacts(model_uri: str) -> str:
        assert model_uri == "models:/test/Production"
        return str(weights_dir)

    monkeypatch.setattr(detectors.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(
        detectors.mlflow.artifacts, "download_artifacts", fake_download_artifacts
    )
    monkeypatch.setattr(detectors, "YOLO", DummyYOLO)

    model = load("models:/test/Production")

    assert isinstance(model, DummyYOLO)
    assert model.weights_path == str(weights_path)


def test_load_uses_cached_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    cache_dir = detectors._model_cache_dir(cache_root, "models:/cached/1")
    cache_dir.mkdir(parents=True)
    cached_weight = cache_dir / "cached.pt"
    cached_weight.write_text("weights")

    def fail_download(_model_uri: str) -> str:
        raise AssertionError("download_artifacts should not be called")

    monkeypatch.setenv("ML_OBJECT_DETECTOR_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(detectors.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(detectors.mlflow.artifacts, "download_artifacts", fail_download)
    monkeypatch.setattr(detectors, "YOLO", DummyYOLO)

    model = load("models:/cached/1")

    assert isinstance(model, DummyYOLO)
    assert model.weights_path == str(cached_weight)


def test_load_caches_downloaded_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    weights_dir = tmp_path / "artifacts"
    weights_dir.mkdir()
    weights_path = weights_dir / "model.pt"
    weights_path.write_text("weights")

    def fake_download_artifacts(model_uri: str) -> str:
        assert model_uri == "models:/cache-me/1"
        return str(weights_dir)

    monkeypatch.setenv("ML_OBJECT_DETECTOR_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(detectors.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(
        detectors.mlflow.artifacts, "download_artifacts", fake_download_artifacts
    )
    monkeypatch.setattr(detectors, "YOLO", DummyYOLO)

    model = load("models:/cache-me/1")

    cache_dir = detectors._model_cache_dir(cache_root, "models:/cache-me/1")
    cached_weight = cache_dir / "model.pt"
    assert isinstance(model, DummyYOLO)
    assert model.weights_path == str(cached_weight)
    assert cached_weight.exists()


def test_load_raises_when_no_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    def fake_download_artifacts(model_uri: str) -> str:
        assert model_uri == "models:/missing/Production"
        return str(empty_dir)

    monkeypatch.setattr(detectors.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(
        detectors.mlflow.artifacts, "download_artifacts", fake_download_artifacts
    )

    with pytest.raises(FileNotFoundError):
        load("models:/missing/Production")


def test_predict_returns_detections() -> None:
    model = DummyModel()
    image = Image.new("RGB", (16, 16), color=(0, 0, 0))

    result = predict(model, image, threshold=0.3)

    assert isinstance(result, DetectionResult)
    assert model.last_conf == 0.3
    assert len(result.detections) == 2
    assert result.detections[0].label == "chip"
