from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

SAMPLE_DIR = Path("experiments/sample_data")


@dataclass(frozen=True)
class DetectionBox:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class InputImage:
    image: Image.Image
    source_label: str


def list_sample_images(sample_dir: Path = SAMPLE_DIR) -> list[Path]:
    if not sample_dir.exists():
        return []
    images = [
        p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(images, key=lambda p: p.name)


def render_image_selector(
    sample_dir: Path = SAMPLE_DIR,
    key_prefix: str = "image",
    show_preview: bool = True,
) -> InputImage | None:
    st.subheader("Image Input")
    source = st.radio(
        "Source",
        ["Sample image", "Upload"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )
    if source == "Sample image":
        samples = list_sample_images(sample_dir)
        if not samples:
            st.warning(f"No samples found under {sample_dir}")
            return None
        selected = st.selectbox(
            "Samples",
            options=samples,
            format_func=lambda p: p.name,
            key=f"{key_prefix}_sample",
        )
        image = Image.open(selected).convert("RGB")
        if show_preview:
            st.image(image, caption=selected.name, use_container_width=True)
        return InputImage(image=image, source_label=str(selected))

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"{key_prefix}_upload",
    )
    if uploaded is None:
        return None
    image = Image.open(uploaded).convert("RGB")
    if show_preview:
        st.image(image, caption=uploaded.name, use_container_width=True)
    return InputImage(image=image, source_label=uploaded.name)


def annotate_image(
    image: Image.Image,
    detections: Iterable[DetectionBox],
    score_threshold: float = 0.0,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for det in detections:
        if det.score < score_threshold:
            continue
        x0, y0, x1, y1 = det.box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        label = f"{det.label} {det.score:.2f}"
        text_box = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(text_box, fill="yellow")
        draw.text((x0, y0), label, fill="black", font=font)
    return annotated


def render_detection_results(
    image: Image.Image,
    detections: list[DetectionBox],
    score_threshold: float = 0.0,
) -> None:
    annotated = annotate_image(image, detections, score_threshold=score_threshold)
    st.image(annotated, use_container_width=True)
    if detections:
        st.dataframe(
            [
                {
                    "label": det.label,
                    "score": round(det.score, 4),
                    "x0": round(det.box[0], 1),
                    "y0": round(det.box[1], 1),
                    "x1": round(det.box[2], 1),
                    "y1": round(det.box[3], 1),
                }
                for det in detections
                if det.score >= score_threshold
            ],
            use_container_width=True,
        )
    else:
        st.info("No detections found.")
