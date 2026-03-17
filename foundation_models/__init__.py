from .clip import CLIP
from .dino import DINO
from .pixels import Pixels
from .siglip2 import SigLIP2

def create_foundation_model(fm_name):
    """
    Create the foundation model given a foundation model name.
    It has the following methods attached to it:
        - fm.embed_img(img)
        - fm.embed_txt(prompts)
    Some foundation models may not have the embed_text method.

    Possible foundation model names:
        - 'clip': CLIP
        - 'dino': DINO
        - 'pixels': Pixels
        - 'siglip2': SigLIP2 (google/siglip2-so400m-patch16-384-jax)
    """
    fm_name = str(fm_name).strip()
    fm_name_lower = fm_name.lower()
    if fm_name_lower == 'clip':
        fm = CLIP()
    elif fm_name_lower == 'dino':
        fm = DINO()
    elif fm_name_lower == 'pixels':
        fm = Pixels()
    elif fm_name_lower == 'siglip2' or fm_name.startswith("google/siglip2"):
        if fm_name.startswith("google/siglip2"):
            model_id = fm_name if fm_name.endswith("-jax") else f"{fm_name}-jax"
        else:
            model_id = "google/siglip2-so400m-patch16-384-jax"
        fm = SigLIP2(model_id=model_id)
    else:
        raise ValueError(f"Unknown foundation model name: {fm_name}")
    return fm
