import torch
import numpy as np
import streamlit as st
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Default model id used in your original script
DEFAULT_DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"

def load_detector(detector_id: str = DEFAULT_DETECTOR_ID, device: str = "cuda"):
    """
    Load and return (processor, model) for Grounding DINO.
    Caller may keep these around to avoid re-loading repeatedly.
    """
    # note: loading large models can be slow and require internet
    processor = AutoProcessor.from_pretrained(detector_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    return processor, model

@st.cache_resource(show_spinner=False)
def detection(image, text_prompt: str,
              detector_id: str = DEFAULT_DETECTOR_ID,
              device: str = "cuda",
              threshold: float = 0.3,
              text_threshold: float = 0.3):
    """
    Run zero-shot detection on PIL image and return boxes as numpy array.
    Returns: numpy array of boxes [[x1,y1,x2,y2], ...]
    """
    processor = AutoProcessor.from_pretrained(detector_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)

    text_labels = [[text_prompt]]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]  # PIL image.size is (w,h) -> target_sizes expects (h,w)
    )
    # results[0]["boxes"] is tensor of boxes in xyxy coords
    boxes = results[0].get("boxes")
    if boxes is None:
        return np.zeros((0,4), dtype=np.int32)
    return boxes.cpu().numpy()