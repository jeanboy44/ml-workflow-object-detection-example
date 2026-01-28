from ml_object_detector.detectors import (
    Detection,
    DetectionResult,
    clear_cache,
    list_models,
    list_model_versions,
    load,
    load_image,
    predict,
)

__all__ = [
    "Detection",
    "DetectionResult",
    "clear_cache",
    "list_models",
    "list_model_versions",
    "load",
    "load_image",
    "predict",
]


def main() -> None:
    models = "\n".join(f"- {name}" for name in list_models())
    print("ml-object-detector: available models")
    print(models)
