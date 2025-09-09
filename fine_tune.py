# fine_tune.py
import torch
from typing import Optional
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def load_inpaint_pipeline(
    model_id: str ,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = torch.float16,
    **pipe_kwargs,
) -> StableDiffusionInpaintPipeline:
    """
    Load and return a Stable Diffusion Inpaint pipeline moved to `device`.
    Use this from your Streamlit app and cache the returned pipeline.
    """
    # Note: for large models, you may need to set `torch_dtype=torch.float16`
    # and ensure your GPU has enough memory. You can also pass cache_dir to
    # StableDiffusionInpaintPipeline.from_pretrained(...) via pipe_kwargs.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        **pipe_kwargs,
    )
    pipe.load_lora_weights("LoRA/amateurphoto-v6-forcu.safetensors", weight_name="amateurphoto-v6-forcu.safetensors")
    pipe.set_adapters("lora", adapter_weights=0.2)
    # move to device
    pipe = pipe.to(device)
    # turn off NSFW safety checker if desired (optional)
    try:
        pipe.safety_checker = None
    except Exception:
        pass
    return pipe

def inpaint_image(
    pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    image: Image.Image,
    mask_image: Image.Image,
    num_inference_steps: int ,
    guidance_scale: float ,
    negative_prompt: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> Image.Image:
    """
    Run inpainting given a loaded pipeline.
    - image: PIL.Image (RGB)
    - mask_image: PIL.Image (L) where white (255) is the area to inpaint, black keep as-is.
    Returns a PIL.Image result.
    """
    # Ensure images are PIL Images
    if not isinstance(image, Image.Image) or not isinstance(mask_image, Image.Image):
        raise ValueError("image and mask_image must be PIL.Image instances")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result