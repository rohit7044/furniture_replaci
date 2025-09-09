# TV-on-Wall — README

A small pipeline and Streamlit demo that detects a wall in a photo, segments the wall region, and places a TV image (warped to real-world dimensions) centered on that wall.

The repo is split into three logical modules:

* `detection.py` — detection functions (Grounding DINO style).
* `segmentation.py` — segmentation functions (SAM2 style).
* `utils.py` — image & geometry helpers, overlay/paste, mask visualization.
* `streamlit_app.py` — Streamlit UI that ties everything together.

---

# Table of contents

* [Installation](#installation)
* [Quick start (run demo)](#quick-start-run-demo)
* [How the pipeline works (high-level)](#how-the-pipeline-works-high-level)
* [File structure](#file-structure)
* [Usage & options](#usage--options)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Example: non-Streamlit quick run](#example-non-streamlit-quick-run)
* [Notes & caveats](#notes--caveats)
* [License & credits](#license--credits)

---

# Installation

> Tested on Python 3.8+. Use a virtual environment.

1. Create & activate a virtualenv (recommended)

```bash
conda create -n tv-wall python=3.10
conda activate tv-wall
```

2. Install PyTorch (match your CUDA / CPU configuration)

Visit the PyTorch install page to pick the right command for your machine. Example CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Install the other packages

Install directly:

```bash
pip install diffusers transformers accelerate scipy safetensors streamlit opencv-python pillow numpy
```
---

# Quick start — run the Streamlit demo

From the project root:

```bash
streamlit run streamlit_app.py
```

This opens a local web UI. Upload:

* a wall photo (photo containing a wall),
* a TV image (frontal/product shot),
* enter the real-world sizes (meters) for the wall and the TV,
* click **Run placement**.

**Notes**

* The first run downloads the detection and segmentation models from Hugging Face — this requires internet and can take time.
* If you have GPU + proper CUDA PyTorch, inference will be much faster.

---

# How the pipeline works (high-level)

1. **Detection** (`detection.py`)
   Uses a Grounding-DINO-like zero-shot detector to find bounding box(es) for the wall region given a text prompt (default: `"a wall"`). Returns pixel bounding boxes.

2. **Segmentation** (`segmentation.py`)
   Crops the detected wall region and passes it to a SAM2-style segmentation model to produce masks. We select the main mask (largest contour).

3. **Utilities & geometry** (`utils.py`)

   * Convert real-world sizes (meters) to pixels using the provided wall dimensions and wall image resolution.
   * Simplify the mask to a 4-corner quadrilateral (or fallback to a min-area rectangle).
   * Compute a centered rectangle with the TV aspect ratio inside the wall quad and warp the TV image into that rectangle using a homography.
   * Composite the warped TV back into the original image and provide intermediate visualizations (mask overlay, warped TV).

4. **Streamlit glue** (`streamlit_app.py`)
   Accepts user inputs (images and sizes), runs detection → segmentation → overlay, and displays intermediate results (detected box, segmentation overlay, warped TV, final composition).

---

# File structure

```
.
├── detection.py
├── segmentation.py
├── utils.py
├── streamlit_app.py
├── requirements.txt    # optional
└── README.md
```

---

# Usage & options

* **Text prompt**: In the Streamlit app you can change the detection prompt from `"a wall"` to something more specific (e.g. `"living room wall"`, `"white wall"`) to improve detection in cluttered scenes.

* **Model selection**: Change `DEFAULT_DETECTOR_ID` / `DEFAULT_SEGMENTER_ID` in `detection.py` / `segmentation.py` if you want different models. Make sure the `transformers` API matches the model architecture.

* **Selecting detection box**: The provided app uses the first detected bounding box. You can adapt `streamlit_app.py` to let the user select which box to use.

* **Adjusting TV placement**: Tweak `utils.shrink_quad_toward_centroid()` or `utils.center_rect_from_minarea()` scale parameters to change how large the TV appears on the wall.

---

# Troubleshooting & tips

* **Very slow / first-run hangs**: model weights download on first use. Be patient or pre-download model caches using the Hugging Face CLI.

* **GPU out of memory**: either switch to CPU (`device="cpu"`), reduce input resolution (downscale the wall image), or use smaller model variants.

* **No detection or wrong detection**: try more specific prompts or crop/resize the input. If the wall is heavily occluded, detection may fail.

* **Segmentation returns no contours**: visualise raw masks with `utils.visualize_masks()` to debug. Make sure the crop actually contains wall pixels.

* **API or model errors**: `transformers` APIs and model availability change over time. If loading `Sam2Processor` / `Sam2Model` or detection classes fails, check model compatibility or swap to a supported model and adjust code.

# Notes & caveats

* The pipeline assumes the wall is a planar rectangle in the scene. Strong lens distortion, non-planar walls, or severe perspective foreshortening will produce imperfect warps.
* Real-world accuracy depends on how accurate the provided wall and TV physical measurements are.
* This is a prototype / research demo — it is not production AR. Use as a starting point and harden model loading, error handling, and UI for production uses.

