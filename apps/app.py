from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import mlflow
import streamlit as st
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)
from ultralytics import YOLO


DETR_MODEL_ID = "facebook/detr-resnet-50"
GROUNDING_DINO_MODEL_DIR = "artifacts/IDEA-Research/grounding-dino-base"
YOLO_PRETRAINED_PATH = "artifacts/yolo/yolo26n.pt"
DEFAULT_MLFLOW_MODEL_URI = "models:/exp05_yolo/Production"


st.set_page_config(page_title="Object Detection Demo", layout="wide")
st.title("Object Detection Demo")
st.write("Upload an image, select a model, and run detection.")
logger.info("Streamlit app started")


@st.cache_resource
def load_detr():
    logger.info("Loading DETR model")
    processor = DetrImageProcessor.from_pretrained(DETR_MODEL_ID, revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(DETR_MODEL_ID, revision="no_timm")
    model.eval()
    return processor, model


@st.cache_resource
def load_grounding_dino():
    logger.info("Loading GroundingDINO model")
    processor = GroundingDinoProcessor.from_pretrained(GROUNDING_DINO_MODEL_DIR)
    model = GroundingDinoForObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_DIR)
    model.eval()
    return processor, model


@st.cache_resource
def load_yolo_pretrained():
    logger.info("Loading YOLO pretrained model")
    return YOLO(YOLO_PRETRAINED_PATH)


@st.cache_resource
def load_yolo_mlflow(model_uri: str):
    logger.info("Loading YOLO model from MLflow: {}", model_uri)
    load_dotenv()
    mlflow.set_tracking_uri("databricks")
    local_dir = mlflow.artifacts.download_artifacts(model_uri)
    weights = list(Path(local_dir).rglob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No .pt weights found under {local_dir}")
    return YOLO(str(weights[0]))


def get_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image, boxes: list[list[float]], labels: list[str], scores: list[float]
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = get_font()
    for box, label, score in zip(boxes, labels, scores, strict=True):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{label} {score:.2f}"
        text_box = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)
    return image


def run_detr(image: Image.Image, threshold: float):
    processor, model = load_detr()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = [model.config.id2label[idx] for idx in results["labels"].tolist()]
    return boxes, labels, scores


def run_yolo(model: YOLO, image_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        results = model(tmp.name)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    scores = results.boxes.conf.cpu().tolist()
    labels = [model.names[int(cls)] for cls in results.boxes.cls.cpu().tolist()]
    return boxes, labels, scores


def run_grounding_dino(image: Image.Image, prompt: str, threshold: float):
    processor, model = load_grounding_dino()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [image.size[::-1]]
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=target_sizes,
    )
    if isinstance(results, list):
        results = results[0]
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = [str(label) for label in results["labels"]]
    return boxes, labels, scores


with st.sidebar:
    model_choice = st.selectbox(
        "Model",
        (
            "DETR (pretrained)",
            "YOLO (pretrained)",
            "GroundingDINO (zero-shot)",
            "YOLO (MLflow registry)",
        ),
    )
    threshold = st.slider("Score threshold", 0.05, 0.95, 0.5, 0.05)
    prompt = "object"
    model_uri = DEFAULT_MLFLOW_MODEL_URI
    if model_choice == "GroundingDINO (zero-shot)":
        prompt = st.text_input("Text prompt", "circle, hole")
    if model_choice == "YOLO (MLflow registry)":
        model_uri = st.text_input("MLflow model URI", DEFAULT_MLFLOW_MODEL_URI)


left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        st.image(input_image, caption="Input image", use_column_width=True)
    else:
        input_image = None
        image_bytes = None
        st.info("Upload an image to start.")

with right:
    run_detection = st.button("Run detection", type="primary", use_container_width=True)
    if run_detection and not input_image:
        st.warning("Please upload an image first.")
    if run_detection and input_image:
        with st.spinner("Running detection..."):
            if model_choice == "DETR (pretrained)":
                logger.info("Running DETR inference")
                boxes, labels, scores = run_detr(input_image.copy(), threshold)
            elif model_choice == "YOLO (pretrained)":
                logger.info("Running YOLO inference (pretrained)")
                yolo_model = load_yolo_pretrained()
                boxes, labels, scores = run_yolo(yolo_model, image_bytes)
            elif model_choice == "YOLO (MLflow registry)":
                logger.info("Running YOLO inference (MLflow)")
                yolo_model = load_yolo_mlflow(model_uri)
                boxes, labels, scores = run_yolo(yolo_model, image_bytes)
            else:
                logger.info("Running GroundingDINO inference")
                boxes, labels, scores = run_grounding_dino(
                    input_image.copy(), prompt, threshold
                )

            output_image = draw_detections(
                input_image.copy(), boxes, labels, scores
            )
            logger.info("Detection complete: {} boxes", len(boxes))
            st.image(output_image, caption="Detection result", use_column_width=True)
            st.write(f"Detections: {len(boxes)}")
