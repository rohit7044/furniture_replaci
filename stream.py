# streamlit_app.py
# -------------------------------------------------------------
# Minimal RGB → Depth Demo using Marigold (diffusers)
# -------------------------------------------------------------
# What this app does (for now)
# 1) Take ONE scene image → predict a depth map with Marigold → visualize
# 2) Take 1–3 object images → predict depth maps → visualize (shown side by side @512x512)
# 3) Original full‑size depth maps are still downloadable as 16‑bit PNGs.

from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Optional imports; we validate availability at runtime
try:
    import torch
    import diffusers
    TORCH_OK = True
except Exception:
    TORCH_OK = False

st.set_page_config(page_title="RGB → Depth (Marigold)", layout="wide")
st.title("RGB → Depth with Marigold (Minimal Demo)")

with st.expander("About this demo"):
    st.markdown(
        """
        This minimal app runs **MarigoldDepthPipeline** on your images and shows:
        - A resized visualization (512×512 per image, side by side for objects)
        - A **downloadable 16‑bit PNG** depth map in original resolution
        
        *Tip:* GPU (CUDA) is faster. On CPU, first run may take longer due to model download.
        """
    )

# ------------------------
# Model loading (cached)
# ------------------------
@st.cache_resource(show_spinner=False)
def load_marigold(device_choice: str = "auto"):
    if not TORCH_OK:
        raise RuntimeError("PyTorch / diffusers not installed. Run: pip install torch diffusers accelerate transformers")

    import torch as _torch
    import diffusers as _diffusers

    device = "cuda" if (device_choice == "auto" and _torch.cuda.is_available()) else (device_choice if device_choice != "auto" else "cpu")
    torch_dtype = _torch.float16 if device == "cuda" else _torch.float32

    pipe = _diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1",
        variant="fp16" if torch_dtype == _torch.float16 else None,
        torch_dtype=torch_dtype,
    ).to(device)
    return pipe, device, torch_dtype


def run_depth(pipe, pil_img: Image.Image):
    """Return (depth_array_float32, vis_pil, depth16_pil)."""
    out = pipe(pil_img)
    pred0 = out.prediction[0]

    # Convert prediction to numpy float32 (robust to either tensor or ndarray)
    if hasattr(pred0, "detach") and hasattr(pred0, "cpu"):
        depth_arr = pred0.detach().cpu().numpy().astype(np.float32)
    else:
        depth_arr = np.array(pred0).astype(np.float32)

    # Normalize for display if needed
    denom = (depth_arr.max() - depth_arr.min()) + 1e-8
    depth_norm = (depth_arr - depth_arr.min()) / denom

    # Use model's image_processor utilities for nice visual & 16-bit export
    vis_list = pipe.image_processor.visualize_depth(out.prediction)
    vis = vis_list[0]
    depth16_list = pipe.image_processor.export_depth_to_16bit_png(out.prediction)
    depth16 = depth16_list[0]

    return depth_norm, vis, depth16

# ---------------------------------------------
# SfM: E-matrix + pose + triangulation
# ---------------------------------------------
def detect_and_match(img1, img2, ratio=0.75):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return pts1, pts2, good

def heuristic_K(img_shape, f_scale=1.2):
    """Heuristic intrinsic matrix if EXIF is unavailable.
    f_px ≈ f_scale * max(W,H), principal point at center.
    """
    h, w = img_shape[:2]
    f = f_scale * max(w, h)
    K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float64)
    return K

def two_view_triangulation(img1, img2, K):
    """Return R, t, 3D points (Nx3), and inlier correspondences."""
    pts1, pts2, good = detect_and_match(img1, img2)
    if len(pts1) < 8:
        return None, None, np.zeros((0, 3))

    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers = inliers.ravel().astype(bool)
    pts1_i, pts2_i = pts1[inliers], pts2[inliers]

    _, R, t, _ = cv2.recoverPose(E, pts1_i, pts2_i, K)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # undistort to normalized coordinates
    pts1n = cv2.undistortPoints(pts1_i.reshape(-1, 1, 2), K, None)
    pts2n = cv2.undistortPoints(pts2_i.reshape(-1, 1, 2), K, None)

    X_h = cv2.triangulatePoints(P1, P2, pts1n.reshape(-1, 2).T, pts2n.reshape(-1, 2).T)
    X = (X_h[:3] / (X_h[3] + 1e-9)).T

    # Keep only points in front of both cameras
    z1 = X[:, 2]
    z2 = (R @ X.T + t).T[:, 2]
    mask = (z1 > 0) & (z2 > 0)
    X = X[mask]

    return R, t, X

def stereo_depth(imgL, imgR, K, baseline=0.1):
    """Compute dense depth from stereo pair."""
    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # StereoSGBM parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # multiple of 16
        blockSize=9,
        P1=8 * 3 * 9**2,
        P2=32 * 3 * 9**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan

    # Depth = f * baseline / disparity
    f = K[0, 0]
    depth = f * baseline / disparity
    return depth, disparity

# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("Settings")
device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0, help="auto → CUDA if available else CPU")

# ------------------------
# UI: Scene image → depth
# ------------------------
col_scene, col_obj = st.columns([1, 1])
with col_scene:
    st.subheader("Step 1: Scene image → Depth")
    scene_file = st.file_uploader("Upload 1-3 SCENE image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if scene_file:
        try:
            with st.spinner("Loading Marigold model..."):
                pipe, device, dtype = load_marigold(device_choice)
            
            vis_list = []
            for i, f in enumerate(scene_file):
                pil_scene = Image.open(f).convert("RGB")
                with st.spinner("Predicting depth (scene)..."):
                    depth_norm, vis, depth16 = run_depth(pipe, pil_scene)
                vis_list.append((pil_scene,vis, depth16, depth_norm, i))
            
            # Show original RGBs side by side
            rgb_concat = np.concatenate([np.array(t[0]) for t in vis_list], axis=1)
            st.image(rgb_concat, caption="Objects ", width="content")

            # Show depth visualizations side by side
            depth_concat = np.concatenate([np.array(t[1]) for t in vis_list], axis=1)
            st.image(depth_concat, caption=f"Objects depth visualization (device={device}, resized)", width="content")

            # Download buttons for full‑size 16‑bit PNGs
            for scene, vis_scene, depth16, depth_norm, idx in vis_list:
                buf = BytesIO()
                depth16.save(buf, format="PNG")
                st.download_button(
                    f"Download object {idx+1} depth (16‑bit PNG, full size)",
                    data=buf.getvalue(),
                    file_name=f"object_{idx+1}_depth_16bit.png",
                    mime="image/png",
                    key=f"sl_obj_{idx}",
                )
                st.caption(f"Object {idx+1} depth stats — min: {float(depth_norm.min()):.3f}, max: {float(depth_norm.max()):.3f}")

            st.caption(f"Depth stats — min: {float(depth_norm.min()):.3f}, max: {float(depth_norm.max()):.3f}")
        except Exception as e:
            st.error(f"Depth prediction failed: {e}")
    

# ------------------------
# UI: Object images → depth
# ------------------------
with col_obj:
    st.subheader("Step 2: Object images → Depth")
    obj_files = st.file_uploader("Upload 1–3 OBJECT images (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if obj_files:
        try:
            with st.spinner("Loading Marigold model..."):
                pipe, device, dtype = load_marigold(device_choice)

            vis_list = []
            for i, f in enumerate(obj_files):
                pil_obj = Image.open(f).convert("RGB")
                with st.spinner(f"Predicting depth (object {i+1})..."):
                    depth_norm, vis, depth16 = run_depth(pipe, pil_obj)
                # Resize for compact display
                vis_resized = vis.resize((512, 512))
                obj_resized = pil_obj.resize((512, 512))
                vis_list.append((obj_resized, vis_resized, depth16, depth_norm, i))

            # Show original RGBs side by side
            rgb_concat = np.concatenate([np.array(t[0]) for t in vis_list], axis=1)
            st.image(rgb_concat, caption="Objects (resized 512x512 each, side by side)", width="content")

            # Show depth visualizations side by side
            depth_concat = np.concatenate([np.array(t[1]) for t in vis_list], axis=1)
            st.image(depth_concat, caption=f"Objects depth visualization (device={device}, resized)", width="content")

            # Download buttons for full‑size 16‑bit PNGs
            for obj_resized, vis_resized, depth16, depth_norm, idx in vis_list:
                buf = BytesIO()
                depth16.save(buf, format="PNG")
                st.download_button(
                    f"Download object {idx+1} depth (16‑bit PNG, full size)",
                    data=buf.getvalue(),
                    file_name=f"object_{idx+1}_depth_16bit.png",
                    mime="image/png",
                    key=f"ol_obj_{idx}",
                )
                st.caption(f"Object {idx+1} depth stats — min: {float(depth_norm.min()):.3f}, max: {float(depth_norm.max()):.3f}")

        except Exception as e:
            st.error(f"Depth prediction failed: {e}")

st.markdown("---")
st.caption("Next steps (when you're ready): add SfM/MVS, point clouds, occupancy grids, and fit checks.")