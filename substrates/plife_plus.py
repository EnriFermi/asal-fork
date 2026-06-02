import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial
import flax.linen as nn


class PLifeNetwork(nn.Module):
    @nn.compact
    def __call__(self, c1, c2): # D, D
        d, = c1.shape
        c = jnp.concatenate([c1, c2], axis=-1)
        x = nn.Dense(features=8)(c)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1+d)(x)
        alpha, dc1 = x[:1], x[1:]
        alpha = jax.nn.tanh(alpha) * 1.5
        dc1 = jax.nn.tanh(dc1)
        return alpha, dc1


# NOTE: the particle space is [0, 1]
"""
Our custom Particle Life ++ substrate.
It allows for the "colors" of the particles to change according to the other particles.
The change dynamics is controlled by a neural network.
"""
class ParticleLifePlus():
    def __init__(self, n_particles=5000, n_colors=6, n_dims=2, x_dist_bins=7,
                 beta=0.3, alpha=0., mass=0.1,
                 dt=0.002, half_life=0.04, rmax=0.1,
                 render_radius=1e-2, sharpness=20.,
                 update_colors=True, world_size=1.,
                 border='torus',
                 neighbor_mode='dense',
                 color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='black'):
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_dims = n_dims
        assert n_dims==2, 'only 2d supported for now'
        self.x_dist_bins = x_dist_bins
        self.plife_net = PLifeNetwork()

        self.render_radius = render_radius
        self.sharpness = sharpness
        self.update_colors = update_colors
        self.world_size = world_size
        self.border = border
        self.color_palette = color_palette
        self.background_color = background_color
        assert border in ('torus', 'wall'), "border must be 'torus' or 'wall'"
        assert neighbor_mode in ('dense', 'radius_exact'), "neighbor_mode must be 'dense' or 'radius_exact'"
        self.neighbor_mode = neighbor_mode

        self.fixed_params = dict(
            beta=jnp.array(beta),
            alpha=None,
            mass=jnp.array(mass),
            dt=jnp.array(dt),
            half_life=jnp.array(half_life),
            rmax=jnp.array(rmax),
        )

    def default_params(self, rng):
        alpha = self.plife_net.init(rng, jnp.zeros((self.n_colors, )), jnp.ones((self.n_colors, )))
        return dict(alpha=alpha)
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)

        c = jax.random.normal(_rng1, (self.n_particles, self.n_colors))
        c = c/jnp.linalg.norm(c, axis=-1, keepdims=True)

        x = jax.random.uniform(_rng2, (self.n_particles, self.n_dims), minval=0., maxval=self.world_size)
        v = jnp.zeros((self.n_particles, self.n_dims))
        return dict(c=c, x=x, v=v)

    def _apply_boundary(self, x, v):
        if self.border == 'torus':
            return x % self.world_size, v

        hit_lo = x < 0.0
        hit_hi = x > self.world_size
        x = jnp.where(hit_lo, -x, x)
        x = jnp.where(hit_hi, 2.0 * self.world_size - x, x)
        v = jnp.where(hit_lo | hit_hi, -v, v)
        x = jnp.clip(x, 0.0, self.world_size)
        return x, v

    def _pairwise_delta(self, x_src, x_tgt):
        r = x_tgt - x_src
        if self.border == 'torus':
            half_world = 0.5 * self.world_size
            r = jax.lax.select(
                r > half_world,
                r - self.world_size,
                jax.lax.select(r < -half_world, r + self.world_size, r),
            )
        return r
    
    def step_state(self, rng, state, params):
        x, v, c = state['x'], state['v'], state['c']

        mass = self.fixed_params['mass']
        half_life = self.fixed_params['half_life']
        dt = self.fixed_params['dt']
        beta = self.fixed_params['beta']
        rmax = self.fixed_params['rmax']

        def force_graph(r, alpha, beta):
            first = r / beta - 1
            second = alpha * (1 - jnp.abs(2 * r - 1 - beta) / (1 - beta))
            cond_first = (r < beta) # (0 <= r) & (r < beta)
            cond_second = (beta < r) & (r < 1)
            return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))
        
        def calc_force_from_delta(r, c1, c2): # force exerted on c1/x1 by c2/x2
            alpha, dc1 = self.plife_net.apply(params['alpha'], c1, c2)
            rlen = jnp.linalg.norm(r)
            rdir = r / (rlen + 1e-8)
            flen = rmax * force_graph(rlen/rmax, alpha, beta)
            force = rdir * flen

            dc1 = dc1 * jax.nn.relu(1.-rlen/rmax)
            return force, dc1 # (n_dims), (n_colors)

        if self.neighbor_mode == 'radius_exact':
            pair_ids = jnp.arange(self.n_particles * self.n_particles)
            force0 = jnp.zeros((self.n_particles, self.n_dims), dtype=x.dtype)
            dc0 = jnp.zeros((self.n_particles, self.n_colors), dtype=c.dtype)

            def add_pair(carry, pair_id):
                force_sum, dc_sum = carry
                i = pair_id // self.n_particles
                j = pair_id - i * self.n_particles
                r = self._pairwise_delta(x[i], x[j])
                rlen = jnp.linalg.norm(r)

                def active(_):
                    return calc_force_from_delta(r, c[i], c[j])

                def inactive(_):
                    return (
                        jnp.zeros((self.n_dims,), dtype=x.dtype),
                        jnp.zeros((self.n_colors,), dtype=c.dtype),
                    )

                force, dc1 = jax.lax.cond(rlen < rmax, active, inactive, operand=None)
                force_sum = force_sum.at[i].add(force)
                dc_sum = dc_sum.at[i].add(dc1)
                return (force_sum, dc_sum), None

            (force_sum, dc1), _ = jax.lax.scan(add_pair, (force0, dc0), pair_ids)
            acc = force_sum / mass
        else:
            r = self._pairwise_delta(x[:, None, :], x[None, :, :])
            f, dc1 = jax.vmap(
                jax.vmap(calc_force_from_delta, in_axes=(0, None, 0)),
                in_axes=(0, 0, None),
            )(r, c, c)
            # f: (this_particle, other_particle, n_dims)
            # dc1: (this_particle, other_particle, n_colors)
            acc = f.sum(axis=-2) / mass
            dc1 = dc1.sum(axis=-2)
        
        mu = (0.5) ** (dt / half_life)
        v = mu * v + acc * dt
        x = x + v * dt
        x, v = self._apply_boundary(x, v)

        if self.update_colors:
            c = c + dc1*dt
            c = c/jnp.linalg.norm(c, axis=-1, keepdims=True)
        return dict(c=c, x=x, v=v)
    
    def render_state(self, state, params, img_size=256):
        background_color = jnp.array(mcolors.to_rgb(self.background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)

        render_radius = self.render_radius
        sharpness = self.sharpness / render_radius

        x, c = state['x'], state['c'][:, :3]
        c = (c+1.)/2.
        mass = jnp.ones((self.n_particles, )) * self.fixed_params['mass']
        # i = jnp.argsort(mass)[::-1]
        # x, c, mass = x[i], c[i], mass[i]

        xgrid = ygrid = jnp.linspace(0, self.world_size, img_size)
        xgrid, ygrid = jnp.meshgrid(xgrid, ygrid, indexing='ij')

        def render_circle(img, circle_data):
            x, y, radius, color = circle_data
            d2 = (x-xgrid)**2 + (y-ygrid)**2
            d = jnp.sqrt(d2)
            # d2 = (d2<radius**2).astype(jnp.float32)[:, :, None]
            # img = d2*color + (1.-d2)*img
            coeff = 1.- (1./(1.+jnp.exp(-sharpness*(d-radius))))
            img = coeff[:, :, None]*color + (1-coeff[:, :, None])*img
            return img, None
    
        radius = jnp.sqrt(mass) * render_radius
        # c = jnp.ones_like(c)
        img, _ = jax.lax.scan(render_circle, img, (*x.T, radius, c))
        return img
    
