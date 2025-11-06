
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split

from .lenia_impl import ConfigLenia
from .lenia_impl import Lenia as LeniaImpl

def inv_sigmoid(x):
    return jnp.log(x) - jnp.log1p(-x)

"""
The Lenia substrate.
The implementation of Lenia is from https://github.com/maxencefaldor/Leniabreeder/tree/main/lenia.
"""
class Lenia():
    def __init__(self, grid_size=128, center_phenotype=True, phenotype_size=100, start_pattern="VT049W", clip1=jnp.inf, clip2=jnp.inf):
        self.grid_size = grid_size
        self.center_phenotype = center_phenotype
        self.phenotype_size = phenotype_size
        self.config_lenia = ConfigLenia(pattern_id=start_pattern, world_size=grid_size)
        self.lenia = LeniaImpl(self.config_lenia)

        self.clip1, self.clip2 = clip1, clip2

        init_carry, init_genotype, other_asset = self.lenia.load_pattern(self.lenia.pattern)
        self.init_carry = init_carry
        self.init_genotype = init_genotype
        self.base_params = inv_sigmoid(self.init_genotype.clip(1e-6, 1.-1e-6))

    def default_params(self, rng):
        return jax.random.normal(rng, self.base_params.shape) * 0.1
    
    def init_state(self, rng, params):
        base_dyn, base_init = self.base_params[:45], self.base_params[45:]
        params_dyn, params_init = params[:45], params[45:]

        params_dyn = jax.nn.sigmoid(base_dyn + jnp.clip(params_dyn, -self.clip1, self.clip1))
        params_init = jax.nn.sigmoid(base_init + jnp.clip(params_init, -self.clip2, self.clip2))
        params = jnp.concatenate([params_dyn, params_init], axis=0)
        # params = jax.nn.sigmoid(jnp.clip(params, -self.clip_genotype, self.clip_genotype)+self.base_params)

        carry = self.lenia.express_genotype(self.init_carry, params)
        state = dict(carry=carry, img=jnp.zeros((self.phenotype_size, self.phenotype_size, 3)))
        # return state
        return self.step_state(rng, state, params) # so init img is not zeros lol
    
    def step_state(self, rng, state, params):
        carry, accum = self.lenia.step(state['carry'], None, phenotype_size=self.phenotype_size, center_phenotype=self.center_phenotype, record_phenotype=True)
        return dict(carry=carry, img=accum.phenotype)
    
    def render_state(self, state, params, img_size=None):
        img = state['img']
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img
