from __future__ import annotations

import streamlit as st
from ml_object_detector import load, predict
from streamlit_utils import DetectionBox, annotate_image, render_image_selector

st.set_page_config(page_title="ML Object Detector", layout="wide")


@st.cache_resource
def load_detector(model_uri: str, tracking_uri: str):
    return load(model_uri, tracking_uri=tracking_uri)


st.title("ML Object Detector")
st.caption("Predictions using the ml-object-detector package")

IMAGE_WIDTH = 420

inputs, original_col, result_col = st.columns([1, 1, 1])

with inputs:
    st.subheader("Inputs")
    registered_model_name = st.text_input(
        "Registered model name",
        value="catalog.schema.model_name",
        help="Format: {catalog}.{schema}.{name}",
    )
    selector_mode = st.radio("Selector", ["Alias", "Version"], horizontal=True)
    if selector_mode == "Alias":
        selector_value = st.text_input(
            "Alias",
            value="production",
            help="Use model alias, e.g. production, champion, or prod",
        )
    else:
        selector_value = st.text_input(
            "Version",
            value="1",
            help="Use a numeric model version",
        )
    tracking_uri = st.text_input("Tracking URI", value="databricks")
    score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.25, 0.05)
    input_image = render_image_selector(
        key_prefix="ml_object_detector", show_preview=False
    )
    run = st.button("Run prediction", type="primary", use_container_width=True)

with original_col:
    st.subheader("Original")
    if input_image:
        st.image(
            input_image.image,
            caption=input_image.source_label,
            width=IMAGE_WIDTH,
        )
    else:
        st.info("Select or upload an image to start.")

with result_col:
    st.subheader("Detections")
    if not input_image:
        st.info("Select or upload an image to start.")
    elif not run:
        st.info("Click Run prediction to see results.")
    else:
        with st.spinner("Running model..."):
            if selector_mode == "Alias":
                model_uri = f"models:/{registered_model_name}@{selector_value}"
            else:
                model_uri = f"models:/{registered_model_name}/{selector_value}"
            model = load_detector(model_uri, tracking_uri)
            result = predict(model, input_image.image, threshold=score_threshold)
            detections = [
                DetectionBox(
                    label=det.label,
                    score=det.score,
                    box=det.box,
                )
                for det in result.detections
            ]
        annotated = annotate_image(
            input_image.image, detections, score_threshold=score_threshold
        )
        st.image(annotated, width=IMAGE_WIDTH)
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
