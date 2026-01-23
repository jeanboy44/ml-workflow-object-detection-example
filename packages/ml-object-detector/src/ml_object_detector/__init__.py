from ml_object_detector.detectors import Detection, DetectionResult, load, load_image, predict

__all__ = [
    "Detection",
    "DetectionResult",
    "load",
    "load_image",
    "predict",
]


def main() -> None:
    print("ml-object-detector: import HuggingFaceDetector or MlflowYoloDetector")
