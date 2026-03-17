import jax
import jax.numpy as jnp
from einops import rearrange


def _resolve_image_size(size_cfg):
    if isinstance(size_cfg, int):
        return int(size_cfg)
    if isinstance(size_cfg, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in size_cfg and size_cfg[key] is not None:
                return int(size_cfg[key])
    return 384


class SigLIP2():
    """
    Fast JAX/Flax SigLIP2 path backed by the official HF JAX checkpoint.
    """

    def __init__(self, model_id="google/siglip2-so400m-patch16-384-jax"):
        try:
            from transformers import AutoProcessor, FlaxSiglipModel
        except ImportError as exc:
            raise ImportError(
                "SigLIP2 JAX requires 'transformers' with Flax SigLIP support installed."
            ) from exc

        self.model_id = str(model_id)
        self.processor_id = (
            self.model_id[:-4] if self.model_id.endswith("-jax") else self.model_id
        )
        self.processor = AutoProcessor.from_pretrained(self.processor_id, use_fast=True)
        self.siglip_model = FlaxSiglipModel.from_pretrained(self.model_id)

        image_processor = self.processor.image_processor
        self.image_size = _resolve_image_size(getattr(image_processor, "size", None))
        self.img_mean = jnp.asarray(image_processor.image_mean, dtype=jnp.float32)
        self.img_std = jnp.asarray(image_processor.image_std, dtype=jnp.float32)

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        h, w, c = img.shape
        if h != self.image_size or w != self.image_size:
            img = jax.image.resize(
                img,
                (self.image_size, self.image_size, c),
                method="bilinear",
            )
        img = rearrange((img - self.img_mean) / self.img_std, "H W C -> 1 C H W")
        z_img = self.siglip_model.get_image_features(pixel_values=img)[0]
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)

    def embed_txt(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        prompts = [str(prompt).lower() for prompt in prompts]
        inputs = self.processor(
            text=prompts,
            return_tensors="jax",
            padding=True,
            truncation=True,
        )
        z_text = self.siglip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )
        return z_text / jnp.linalg.norm(z_text, axis=-1, keepdims=True)
