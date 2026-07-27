import jax
import jax.numpy as jnp
from functools import partial

class ReintegrationTracking:

    #-------------------------------------------------------------------

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, border="wall", has_hidden=False, 
                 mix="stoch"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix

    #-------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        
        if self.has_hidden:
            return self._apply_with_hidden(*args, **kwargs)
        else:
            return self._apply_without_hidden(*args, **kwargs)

    #-------------------------------------------------------------------

    def _bilinear_index_weights(self, points: jax.Array):
        """
        Precompute bilinear stencil indices and weights for point coordinates.
        points: (N,2) in center coordinates (i+0.5).
        """
        fy = points[:, 0] - 0.5
        fx = points[:, 1] - 0.5

        if self.border == "torus":
            fy = jnp.mod(fy, self.SX)
            fx = jnp.mod(fx, self.SY)
            i0 = jnp.floor(fy).astype(jnp.int32)
            j0 = jnp.floor(fx).astype(jnp.int32)
            i1 = (i0 + 1) % self.SX
            j1 = (j0 + 1) % self.SY
        else:
            fy = jnp.clip(fy, 0.0, self.SX - 1.0)
            fx = jnp.clip(fx, 0.0, self.SY - 1.0)
            i0 = jnp.floor(fy).astype(jnp.int32)
            j0 = jnp.floor(fx).astype(jnp.int32)
            i1 = jnp.clip(i0 + 1, 0, self.SX - 1)
            j1 = jnp.clip(j0 + 1, 0, self.SY - 1)

        wy = fy - i0.astype(fy.dtype)
        wx = fx - j0.astype(fx.dtype)
        return i0, j0, i1, j1, wy, wx

    #-------------------------------------------------------------------

    def _sample_tensor_bilinear(self, tensor: jax.Array, points: jax.Array) -> jax.Array:
        """
        Bilinear sample tensor (SX,SY,...) at points (N,2) -> (N,...).
        """
        i0, j0, i1, j1, wy, wx = self._bilinear_index_weights(points)
        t00 = tensor[i0, j0]
        t10 = tensor[i1, j0]
        t01 = tensor[i0, j1]
        t11 = tensor[i1, j1]

        w00 = (1.0 - wy) * (1.0 - wx)
        w10 = wy * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w11 = wy * wx
        for _ in range(t00.ndim - 1):
            w00 = w00[..., None]
            w10 = w10[..., None]
            w01 = w01[..., None]
            w11 = w11[..., None]
        return w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11

    def _to_flow_2d(self, F: jax.Array, A: jax.Array | None = None, channel: int = -1, reduce: str = "mass_weighted") -> jax.Array:
        """
        Convert F to a 2D vector field (SX,SY,2) for point advection.
        F can be (SX,SY,2) or (SX,SY,2,C).
        """
        if F.ndim == 3:
            return F
        if F.ndim != 4:
            raise ValueError(f"Expected F with ndim 3 or 4, got shape={F.shape}.")

        if channel is not None and int(channel) >= 0:
            c = jnp.clip(jnp.asarray(channel, dtype=jnp.int32), 0, F.shape[-1] - 1)
            return F[..., c]

        if reduce == "mean":
            return jnp.mean(F, axis=-1)

        if reduce == "mass_weighted" and A is not None:
            w = jnp.clip(A, 0.0, jnp.inf)
            den = jnp.sum(w, axis=-1, keepdims=True)
            return jnp.sum(F * w[:, :, None, :], axis=-1) / (den + 1e-8)

        return jnp.mean(F, axis=-1)

    #-------------------------------------------------------------------

    def _sample_flow_bilinear(self, flow: jax.Array, points: jax.Array) -> jax.Array:
        """
        Bilinear sample of (SX,SY,2) flow at point coordinates (N,2), where
        points use FlowLenia coordinates with cell centers at i+0.5.
        """
        return self._sample_tensor_bilinear(flow, points)

    #-------------------------------------------------------------------

    def _sample_channels_bilinear(self, A: jax.Array, points: jax.Array) -> jax.Array:
        """
        Bilinear sample A (SX,SY,C) at points -> (N,C).
        """
        return self._sample_tensor_bilinear(A, points)

    #-------------------------------------------------------------------

    def sample_point_channels(self, points: jax.Array, A: jax.Array, key: jax.Array) -> jax.Array:
        """
        Sample channel id per point from local (bilinear) A-proportional probabilities.
        Returns int32 channel ids of shape (N,).
        """
        w = jnp.clip(self._sample_channels_bilinear(A, points), 0.0, jnp.inf)
        den = jnp.sum(w, axis=-1, keepdims=True)
        c = w.shape[-1]
        uniform = jnp.full_like(w, 1.0 / jnp.maximum(1, c))
        probs = jnp.where(den > 1e-8, w / (den + 1e-8), uniform)
        logits = jnp.log(jnp.clip(probs, 1e-8, 1.0))
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    #-------------------------------------------------------------------

    def _flow_for_points(
        self,
        points: jax.Array,
        F: jax.Array,
        A: jax.Array | None = None,
        channel: int = -1,
        reduce: str = "mass_weighted",
        point_channels: jax.Array | None = None,
    ) -> jax.Array:
        """
        Return flow sampled at points as (N,2), with optional per-particle channel ids.
        """
        if F.ndim == 3:
            return self._sample_flow_bilinear(F, points)
        if F.ndim != 4:
            raise ValueError(f"Expected F with ndim 3 or 4, got shape={F.shape}.")

        f_pts = self._sample_tensor_bilinear(F, points)  # (N,2,C)

        if point_channels is not None:
            ch = jnp.clip(point_channels.astype(jnp.int32), 0, F.shape[-1] - 1)
            idx = ch[:, None, None]
            return jnp.take_along_axis(f_pts, idx, axis=-1)[..., 0]

        if channel is not None and int(channel) >= 0:
            c = int(channel)
            c = max(0, min(c, int(F.shape[-1]) - 1))
            return f_pts[..., c]

        if reduce == "mean":
            return jnp.mean(f_pts, axis=-1)

        if reduce == "mass_weighted" and A is not None:
            w = jnp.clip(self._sample_channels_bilinear(A, points), 0.0, jnp.inf)  # (N,C)
            den = jnp.sum(w, axis=-1, keepdims=True)  # (N,1)
            return jnp.sum(f_pts * w[:, None, :], axis=-1) / (den + 1e-8)

        return jnp.mean(f_pts, axis=-1)

    #-------------------------------------------------------------------

    def _apply_point_border(self, pts: jax.Array) -> jax.Array:
        if self.border == "torus":
            yy = jnp.mod(pts[:, 0] - 0.5, self.SX) + 0.5
            xx = jnp.mod(pts[:, 1] - 0.5, self.SY) + 0.5
            return jnp.stack((yy, xx), axis=-1)

        lo = jnp.array([self.sigma, self.sigma], dtype=pts.dtype)
        hi = jnp.array([self.SX - self.sigma, self.SY - self.sigma], dtype=pts.dtype)
        return jnp.clip(pts, lo, hi)

    #-------------------------------------------------------------------

    def advect_particles(
        self,
        points: jax.Array,
        F: jax.Array,
        A: jax.Array | None = None,
        channel: int = -1,
        reduce: str = "mass_weighted",
        point_channels: jax.Array | None = None,
        channel_mode: str = "mix",  # "mix" | "fixed" | "resample"
        key: jax.Array | None = None,
        noise_model: str = "none",  # "none" | "rt_box" | "gaussian"
        diffusion_scale: float = 1.0,
    ):
        """
        Advect particles with optional stochastic diffusion and channel-id tracking.

        Returns:
            points_next: (N,2)
            channels_next: (N,) int32 or None
        """
        ch_mode = str(channel_mode)
        channels_next = point_channels

        if F.ndim == 4 and ch_mode == "resample":
            if key is None:
                raise ValueError("channel_mode='resample' requires key.")
            key, kch = jax.random.split(key)
            if A is None:
                raise ValueError("channel_mode='resample' requires A.")
            channels_next = self.sample_point_channels(points, A, kch)
        elif ch_mode == "fixed":
            if F.ndim == 4 and point_channels is None:
                raise ValueError("channel_mode='fixed' requires point_channels.")
            channels_next = point_channels
        elif ch_mode != "mix":
            raise ValueError(f"Unknown channel_mode={channel_mode!r}.")

        v = self._flow_for_points(
            points=points,
            F=F,
            A=A,
            channel=channel,
            reduce=reduce,
            point_channels=channels_next if ch_mode in ("fixed", "resample") else None,
        )

        ma = self.dd - self.sigma
        delta = jnp.clip(self.dt * v, -ma, ma)
        pts = points + delta

        nm = str(noise_model)
        ds = jnp.asarray(diffusion_scale, dtype=pts.dtype)
        if nm == "rt_box":
            if key is None:
                raise ValueError("noise_model='rt_box' requires key.")
            key, kn = jax.random.split(key)
            eps = jax.random.uniform(kn, shape=pts.shape, minval=-self.sigma, maxval=self.sigma, dtype=pts.dtype)
            pts = pts + ds * eps
        elif nm == "gaussian":
            if key is None:
                raise ValueError("noise_model='gaussian' requires key.")
            key, kn = jax.random.split(key)
            # Match variance with Uniform[-sigma, sigma] when diffusion_scale=1.
            std = ds * (self.sigma / jnp.sqrt(jnp.asarray(3.0, dtype=pts.dtype)))
            eps = jax.random.normal(kn, shape=pts.shape, dtype=pts.dtype) * std
            pts = pts + eps
        elif nm != "none":
            raise ValueError(f"Unknown noise_model={noise_model!r}.")

        pts = self._apply_point_border(pts)
        return pts, channels_next

    #-------------------------------------------------------------------

    def advect_points(
        self,
        points: jax.Array,
        F: jax.Array,
        A: jax.Array | None = None,
        channel: int = -1,
        reduce: str = "mass_weighted",
    ) -> jax.Array:
        """
        Advect explicit Lagrangian points by one FlowLenia step using the same
        dt/dd/sigma/border constraints as reintegration.

        Args:
            points: (N,2) in FlowLenia coordinates (cell centers are i+0.5).
            F: (SX,SY,2) or (SX,SY,2,C) flow field.
            A: optional (SX,SY,C) activations for mass-weighted channel mixing.
            channel: if >=0, use this F channel directly.
            reduce: "mass_weighted" or "mean" when channel < 0.
        """
        pts, _ = self.advect_particles(
            points=points,
            F=F,
            A=A,
            channel=channel,
            reduce=reduce,
            point_channels=None,
            channel_mode="mix",
            key=None,
            noise_model="none",
            diffusion_scale=1.0,
        )
        return pts

    #-------------------------------------------------------------------

    def _apply_without_hidden(
        self,
        A: jax.Array,
        F: jax.Array,
        categorical_gumbel: jax.Array | None = None,
    )->jax.Array:
        if categorical_gumbel is not None:
            raise ValueError(
                "categorical_gumbel requires hidden-state reintegration"
            )

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def step(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)
            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)

        nA = step(A, mu, dxs, dys).sum(0)
        
        return nA

    #-------------------------------------------------------------------

    def _apply_with_hidden(
        self,
        A: jax.Array,
        H: jax.Array,
        F: jax.Array,
        categorical_gumbel: jax.Array | None = None,
    ):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)
        
        @partial(jax.vmap, in_axes = (None, None, None, 0, 0))
        def step_flow(A, H, mu, dx, dy):
            """Summary
            """
            Ar = jnp.roll(A, (dx, dy), axis = (0, 1))
            Hr = jnp.roll(H, (dx, dy), axis = (0, 1)) #(x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis = (0, 1))

            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)

            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA, Hr

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
        nA, nH = step_flow(A, H, mu, dxs, dys)

        if self.mix == 'avg':
            nH = jnp.sum(nH * nA.sum(axis = -1, keepdims = True), axis = 0)  
            nA = jnp.sum(nH, axis = 0)
            nH = nH / (nA.sum(axis = -1, keepdims = True)+1e-10)

        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis = -1, keepdims = True)) - 1
            nA = jnp.sum(nA, axis = 0)
            nH = jnp.sum(nH * expnA, axis = 0) / (expnA.sum(axis = 0)+1e-10) #avg rule

        elif self.mix == "stoch":
            logits = jnp.log(nA.sum(axis=-1, keepdims=True))
            if categorical_gumbel is None:
                categorical = jax.random.categorical(
                    jax.random.PRNGKey(42),
                    logits,
                    axis=0,
                )
            else:
                if categorical_gumbel.shape != logits.shape:
                    raise ValueError(
                        "categorical_gumbel shape must match stochastic RT "
                        f"logits: {categorical_gumbel.shape} != {logits.shape}"
                    )
                categorical = jnp.argmax(
                    categorical_gumbel + logits,
                    axis=0,
                )
            mask=jax.nn.one_hot(categorical,num_classes=(2*self.dd+1)**2,axis=-1)
            mask=jnp.transpose(mask,(3,0,1,2)) 
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)

        elif self.mix == "stoch_gene_wise":
            mask = jnp.concatenate(
              [jax.nn.one_hot(jax.random.categorical(
                                                    jax.random.PRNGKey(42), 
                                                    jnp.log(nA.sum(axis = -1, keepdims = True)), 
                                                    axis=0),
                              num_classes=(2*dd+1)**2,axis=-1)
              for _ in range(H.shape[-1])], 
              axis = 2)
            mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)
        
        return nA, nH

    #-------------------------------------------------------------------
