from __future__ import annotations

import time

import streamlit as st
from ml_object_detector import list_model_versions, list_models, load,predict
from streamlit_utils import DetectionBox, annotate_image, render_image_selector

st.set_page_config(page_title="ML Object Detector", layout="wide")

st.markdown(
    """
    <style>
    .inputs-panel {
        border: 1px solid #d0d4dd;
        border-radius: 12px;
        padding: 16px;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(
    model_name: str,
    tracking_uri: str,
    model_version: str | None,
    model_alias: str | None,
):
    return load(
        model_name,
        tracking_uri=tracking_uri,
        model_version=model_version,
        model_alias=model_alias,
    )


st.title("ML Object Detector")
st.caption("Predictions using the ml-object-detector package")

IMAGE_WIDTH = 420

inputs, original_col, result_col = st.columns([1, 1, 1])

with inputs.container(border=True):
    st.subheader("Inputs")
    model_options = list_models()
    if not model_options:
        st.warning("No registered models available.")
        registered_model_name = ""
    else:
        registered_model_name = st.selectbox(
            "Registered model name",
            options=model_options,
            index=0,
            help="Only registered models are available",
        )
    tracking_uri = st.text_input("Tracking URI", value="databricks")
    selector_mode = st.radio("Selector", ["Alias", "Version"], horizontal=True)
    if selector_mode == "Alias":
        selector_value = st.text_input(
            "Alias",
            value="",
            help="Use model alias registered in Unity Catalog (e.g. champion, prod)",
        )
    else:
        version_options = (
            list_model_versions(registered_model_name, tracking_uri)
            if registered_model_name
            else []
        )
        if version_options:
            selector_value = st.selectbox(
                "Version",
                options=version_options,
                index=len(version_options) - 1,
                help="Use a numeric model version",
            )
        else:
            st.warning("No versions found for the selected model.")
            selector_value = ""
    score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.25, 0.05)
    input_image = render_image_selector(
        key_prefix="ml_object_detector", show_preview=False
    )
    selector_value = selector_value or ""
    selector_valid = bool(selector_value.strip())
    run = st.button(
        "Run prediction",
        type="primary",
        use_container_width=True,
        disabled=not model_options or not selector_valid,
    )
    if selector_mode == "Alias" and not selector_valid:
        st.info("Alias를 입력해야 실행할 수 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

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
            start_time = time.perf_counter()
            if selector_mode == "Alias":
                model = load_detector(
                    registered_model_name,
                    tracking_uri,
                    model_version=None,
                    model_alias=selector_value,
                )
            else:
                model = load_detector(
                    registered_model_name,
                    tracking_uri,
                    model_version=selector_value,
                    model_alias=None,
                )
            result = predict(model, input_image.image, threshold=score_threshold)
            detections = [
                DetectionBox(
                    label=det.label,
                    score=det.score,
                    box=det.box,
                )
                for det in result.detections
            ]
            elapsed = time.perf_counter() - start_time
        annotated = annotate_image(
            input_image.image, detections, score_threshold=score_threshold
        )
        st.image(annotated, width=IMAGE_WIDTH)
        st.caption(f"Inference time: {elapsed:.3f}s")
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
