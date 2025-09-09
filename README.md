
---

# TV-on-Wall — README

A small pipeline and Streamlit demo that detects a wall in a photo, segments the wall region, and places a TV image (warped to real-world dimensions) centered on that wall.
Optionally, the pipeline can refine the result with Stable Diffusion inpainting to make the TV placement look more realistic.

The repo is split into four logical modules:

* `detection.py` — detection functions (Grounding DINO).
* `segmentation.py` — segmentation functions (SAM2).
* `utils.py` — image & geometry helpers, overlay/paste, mask visualization.
* `fine_tune.py` — Stable Diffusion inpainting pipeline wrapper.
* `streamlit_app.py` — Streamlit UI that ties everything together.

---

# Table of contents

* [Installation](#installation)
* [Quick start (run demo)](#quick-start-run-demo)
* [How the pipeline works](#how-the-pipeline-works)
* [File structure](#file-structure)
* [Usage & options](#usage--options)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Notes & caveats](#notes--caveats)
* [License & credits](#license--credits)

---

# Installation

> Tested on Python 3.10+. Use a virtual environment.

1. Create & activate a virtualenv (recommended):

```bash
conda create -n tv-wall python=3.10
conda activate tv-wall
```

2. Install PyTorch (match your CUDA / CPU configuration):

Check the [PyTorch install page](https://pytorch.org/get-started/locally/) to pick the right command. Example for CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Install the other packages:

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

* a **wall photo** (photo containing a wall),
* a **TV image** (front-facing PNG/JPG),
* enter the **real-world sizes** (meters) for the wall and the TV,
* click **Run placement**.

**Notes:**

* The first run downloads the detector, segmenter, and inpainting models from Hugging Face — this requires internet and may take time.
* If you have GPU + CUDA PyTorch, inference will be much faster.

---

# How the pipeline works

1. **Detection** (`detection.py`)
   Uses Grounding DINO to find bounding box(es) for the wall region given a text prompt (default: `"a wall"`).

2. **Segmentation** (`segmentation.py`)
   Crops the detected wall region and runs SAM2 to produce a wall mask. The largest mask is chosen.

3. **Geometry & overlay** (`utils.py`)

   * Convert real-world sizes (meters) → pixels using wall dimensions and image resolution.
   * Simplify the wall mask to 4 corners.
   * Warp the TV image to a centered rectangle with correct aspect ratio inside the wall quad (homography).
   * Composite the warped TV back into the original wall image.

4. **Optional realism enhancement** (`fine_tune.py`)

   * Wraps a Stable Diffusion inpainting pipeline (`stabilityai/stable-diffusion-2-inpainting`).
   * Runs one inpaint pass with a narrow border mask around the TV → generates soft shadows / blends edges.

5. **Streamlit glue** (`streamlit_app.py`)
   Handles file upload, user inputs, runs the pipeline, and displays results (detected box, mask overlay, warped TV, final composition, optional refined output).

---

# File structure

```
.
├── detection.py
├── segmentation.py
├── utils.py
├── fine_tune.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# Usage & options

* **Text prompt** — Adjust in the Streamlit app. Default: `"a wall"`. Try `"living room wall"` or `"white wall"` for cluttered scenes.

* **Model IDs** — Change in `detection.py` / `segmentation.py` if you want different Hugging Face models.

* **TV placement** — The TV is centered on the wall by default. Adjust scale fraction inside `utils.py` if you want smaller/larger placement.

* **Inpainting realism** — In the Streamlit app you can toggle “Enhance realism with Stable Diffusion inpainting.” This uses a border mask so only edges get refined, not the TV interior.

---

# Troubleshooting & tips

* **First run is slow** — models download automatically. Use `HF_HOME` env var to cache in a specific folder if needed.

* **GPU memory errors** — lower wall image resolution, reduce inference steps, or run on CPU (`device="cpu"`).

* **Detection misses wall** — try a different text prompt or crop the image tighter.

* **Segmentation fails** — visualize masks (`utils.visualize_masks`) to debug; sometimes SAM needs a clearer input.

* **TV gets distorted in inpainting** — ensure the inpaint mask is only a thin border, not the full TV region.

---

# Notes & caveats

* Assumes the wall is approximately planar. Strong perspective, fisheye, or occlusion can break placement.
* Real-world scaling depends on accurate wall and TV measurements.
* Inpainting refinement is limited: it adds shadows/blending, but won’t always be photoreal in all lighting conditions.
* This is a prototype / research demo, not production-ready AR.

---

# License & credits

* [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* [SAM2](https://ai.meta.com/sam/)
* [Stable Diffusion Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
* Hugging Face 🤗 `transformers` + `diffusers`

---