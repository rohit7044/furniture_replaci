# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Import your modularized functions
from detection import detection
from segmentation import segmentation
import utils

st.set_page_config(layout="wide", page_title="TV-on-Wall Layout")

st.title("Place TV on Wall — demo")
st.write("Upload a wall image and a TV image, provide real-world sizes (meters), then press Run.")

col1, col2 = st.columns(2)

with col1:
    wall_file = st.file_uploader("Wall image (photo)", type=["jpg","jpeg","png"], key="wall")
    wall_width_m = st.number_input("Wall width (meters)", min_value=0.1, value=3.8, step=0.1)
    wall_height_m = st.number_input("Wall height (meters)", min_value=0.1, value=2.6, step=0.1)

with col2:
    tv_file = st.file_uploader("TV image", type=["jpg","jpeg","png"], key="tv")
    tv_width_m = st.number_input("TV width (meters)", min_value=0.01, value=1.2, step=0.01)
    tv_height_m = st.number_input("TV height (meters)", min_value=0.01, value=0.7, step=0.01)

text_prompt = st.text_input("Text prompt for detector", value="a wall")

run = st.button("Run placement")

# Helper to convert uploaded file to PIL
def load_pil(uploader_file):
    if uploader_file is None:
        return None
    return Image.open(io.BytesIO(uploader_file.getvalue())).convert("RGB")

if run:
    if wall_file is None or tv_file is None:
        st.error("Please upload both wall and TV images.")
    else:
        wall_pil = load_pil(wall_file)
        tv_pil = load_pil(tv_file)

        st.sidebar.info("Running detection (Grounding DINO) and segmentation (SAM2). Model download may take time on first run.")

        # Convert TV to numpy BGR for cv2 ops (utils functions expect numpy arrays)
        tv_np = np.array(tv_pil)[:, :, ::-1].copy()  # RGB->BGR

        # 1) Detection
        with st.spinner("Running detection..."):
            boxes = detection(wall_pil, text_prompt)
        if boxes.shape[0] == 0:
            st.warning("No boxes detected with the provided prompt.")
            st.stop()

        # Show first detection box on image
        box0 = boxes[0]
        wall_with_box = utils.drawbbox(np.array(wall_pil), [box0])  # expects BGR
        wall_display = cv2.cvtColor(wall_with_box, cv2.COLOR_BGR2RGB)
        st.subheader("Detected bounding box (first)")
        st.image(wall_display, width = "content")

        # 2) Crop to bbox and run segmentation
        cropped = utils.crop_bbox(wall_pil, box0)
        with st.spinner("Running segmentation (SAM2)..."):
            seg_masks = segmentation(cropped)

        # 3) Visualize mask over crop
        vis_masks = utils.visualize_masks(cropped, seg_masks)
        vis_masks_rgb = cv2.cvtColor(vis_masks, cv2.COLOR_BGR2RGB)
        st.subheader("Segmentation overlay (on cropped detection)")
        st.image(vis_masks_rgb, width = "content")

        # 4) Get wall corners from the segmentation mask
        try:
            wall_corners = utils.get_wall_corners(seg_masks, orig_image=cropped)
        except Exception as e:
            st.error(f"Failed to get wall corners from segmentation: {e}")
            st.stop()

        # 5) Overlay TV onto the cropped wall (warped)
        # For overlay function we expect wall_img (numpy BGR), tv_img (numpy BGR)
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        warped_tv,croppedwall_withTV, mask = utils.overlay_tv_fixed_size_on_wall(
            cropped_bgr, tv_np, wall_width_m, wall_height_m, tv_width_m, tv_height_m, wall_corners
        )

        warped_rgb = cv2.cvtColor(croppedwall_withTV, cv2.COLOR_BGR2RGB)
        st.subheader("Warped TV (on cropped wall)")
        st.image(warped_rgb, width = "content")

        # 6) Paste warped into original (full) image
        orig_bgr = cv2.cvtColor(np.array(wall_pil), cv2.COLOR_RGB2BGR)
        composited = utils.paste_warped_into_original(orig_bgr, warped_tv, mask, box0)
        composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
        st.subheader("Final composition on original image")
        st.image(composited_rgb, width = "content")

        st.success("Done!")
