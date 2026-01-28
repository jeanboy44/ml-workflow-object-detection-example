from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path("artifacts/onnx/yolo26n.onnx")
IMAGE_PATH = Path("experiments/sample_data/cat_01.jpg")


def _closest_stride_size(size: int, stride: int = 32) -> int:
    return max(stride, (size // stride) * stride)


def preprocess_image(image: Image.Image, input_shape: tuple[int, int]) -> np.ndarray:
    resized = image.resize(input_shape)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}. Export the model first."
        )
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    image = Image.open(IMAGE_PATH).convert("RGB")
    if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        height = input_shape[2]
        width = input_shape[3]
    else:
        height = _closest_stride_size(640)
        width = _closest_stride_size(640)
    input_tensor = preprocess_image(image, (width, height))

    outputs = session.run(None, {input_name: input_tensor})
    for idx, output in enumerate(outputs):
        print(f"Output[{idx}] shape: {np.asarray(output).shape}")


if __name__ == "__main__":
    main()
