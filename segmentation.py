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

def load_segmenter(segmenter_id, device):
    """
    Load and return (processor, model) for SAM2.
    """
    processor = Sam2Processor.from_pretrained(segmenter_id)
    model = Sam2Model.from_pretrained(segmenter_id).to(device)
    return processor, model

def segmentation(image, processor=None, model=None,
                 segmenter_id: str = DEFAULT_SEGMENTER_ID, device: str = "cuda"):
    """
    If processor/model provided, use those. Otherwise load internally.
    Returns segmentation masks (post-processed).
    """
    if processor is None or model is None:
        processor = Sam2Processor.from_pretrained(segmenter_id)
        model = Sam2Model.from_pretrained(segmenter_id).to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    return masks
