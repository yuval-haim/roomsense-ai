from __future__ import annotations

from pathlib import Path
import json
import streamlit as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline

st.set_page_config(page_title="RoomSense AI", layout="wide")
st.title("RoomSense AI - Indoor Safety & Layout Inspector")
st.write("Run the reproducible annotation-backed demo or use Grounding DINO for arbitrary room images.")

backend = st.sidebar.selectbox("Detector backend", ["annotation", "grounding-dino"])
st.sidebar.caption("Grounding DINO requires `pip install -r requirements-grounding.txt`.")
box_threshold = st.sidebar.slider("Grounding DINO box threshold", 0.05, 0.8, 0.25, 0.05)
text_threshold = st.sidebar.slider("Grounding DINO text threshold", 0.05, 0.8, 0.20, 0.05)

images = sorted((ROOT / "data/demo_images").glob("*.jpg"))
choice = st.selectbox("Bundled demo image", images, format_func=lambda p: p.name)
uploaded = st.file_uploader("Or upload a room photo", type=["jpg", "jpeg", "png"])

image_path = choice
if uploaded is not None:
    image_path = ROOT / "outputs" / "uploads" / uploaded.name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(uploaded.getbuffer())

if st.button("Analyze room"):
    try:
        report = RoomSensePipeline(
            ROOT / "data/annotations",
            detector_backend=backend,
            prompts_path=ROOT / "configs/object_prompts.yaml",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        ).analyze(image_path, ROOT / "outputs/examples")
        overlay = ROOT / "outputs/examples" / f"{image_path.stem}_overlay.jpg"
        depth = ROOT / "outputs/examples" / f"{image_path.stem}_depth.png"
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Annotated image")
            st.image(str(overlay), use_container_width=True)
        with c2:
            st.subheader("Depth prior")
            st.image(str(depth), use_container_width=True)
        st.subheader("Structured report")
        st.json(json.loads(report.model_dump_json()))
    except Exception as exc:
        st.error(str(exc))
        st.info("For uploaded images, choose the Grounding DINO backend and install the optional model requirements.")
