import streamlit as st
import torch
from transformers import Sam2Processor, Sam2Model

# Default segmenter id from original script
DEFAULT_SEGMENTER_ID = "facebook/sam2.1-hiera-tiny"

def load_segmenter(segmenter_id: str = DEFAULT_SEGMENTER_ID, device: str = "cuda"):
    """
    Load and return (processor, model) for SAM2.
    """
    processor = Sam2Processor.from_pretrained(segmenter_id)
    model = Sam2Model.from_pretrained(segmenter_id).to(device)
    return processor, model

@st.cache_resource(show_spinner=False)
def segmentation(image, segmenter_id: str = DEFAULT_SEGMENTER_ID, device: str = "cuda"):
    """
    Run SAM2 segmentation on a PIL image. Returns masks in the processed format
    (post-processed via processor.post_process_masks), same as your earlier code.
    The return is a list or tensor suitable for get_wall_corners in utils.py.
    """
    seg_model = Sam2Model.from_pretrained(segmenter_id).to(device)
    seg_processor = Sam2Processor.from_pretrained(segmenter_id)

    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)

    masks = seg_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    # masks is typically a tensor or list representing predicted masks for the image
    return masks
