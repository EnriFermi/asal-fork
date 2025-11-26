import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.random import split
from .utils import conn_from_matrix, get_kernels_fft, sobel, growth
from .reintegration_tracking import ReintegrationTracking
from .lenia_flow_impl import Config


def inv_sigmoid(x):
    return jnp.log(x) - jnp.log1p(-x)


class FlowLenia:
    """
    FlowLenia substrate wrapper matching the ASAL substrate interface.

    - default_params(rng): returns an unconstrained offset tree around the base
      parameters (in logit space) for R, r, m, s, a, b, w.
    - init_state(rng, params): builds the model with decoded params, computes
      kernels via FlowLeniaParams.initialize, and seeds a small central patch
      in A and P to kick-start dynamics.
    - step_state(rng, state, params): advances one simulation step.
    - render_state(state, params, img_size): renders RGB image from state.
    """

    def __init__(
        self,
        grid_size: int = 128,
        C: int = 1,
        k: int = 10,
        M: jnp.ndarray = jnp.array([
                [2, 1, 0],
                [0, 2, 1],
                [1, 0, 2]]),
        dd: int = 5,
        dt: float = 0.2,
        sigma: float = 0.65,
        border: str = "wall",
        mix_rule: str = "stoch",
        seed_patch_size: int = 40,
        seed_n_patches: int = 1,
        seed_mode: str = "notebook_centers",  # 'center' | 'random_patches' | 'notebook_centers'
        p_constant_per_patch: bool = True,
        render_mode: str = "Pcolor",  # 'A' | 'Pcolor'
        clip1: float = float("inf"),
        clip2: float = float("inf"),
        # mutation controls (optional)
        mutation: bool = False,
        mutation_patch_size: int = 20,
        mutation_prob: float = 0.1,
        # food/resource mechanics (optional)
        food_enabled: bool = False,
        food_spawn_interval: int = 128,
        food_n_patches: int = 3,
        food_patch_size: int = 16,
        food_amount: float = 1.0,
        food_consume_rate: float = 0.05,
        food_bonus: float = 1.0,
        mass_decay: float = 0.0,
        food_green_channel: int = 1,
        food_auto_size: bool = False,
        food_conv_mode: str = "scalar",  # 'scalar' | 'conv'
        food_vis_scale: float = 1.0,
        food_vis_color=(0.6, 0.3, 0.0),  # RGB overlay for food
        food_diffusion_alpha: float = 0.0,  # blend factor for food diffusion (0=off)
    ):
        self.grid_size = grid_size
        self.C = C
        self.k = k
        self.M = M
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.border = border
        self.mix_rule = mix_rule
        self.seed_patch_size = seed_patch_size
        self.seed_n_patches = seed_n_patches
        self.seed_mode = seed_mode
        self.p_constant_per_patch = bool(p_constant_per_patch)
        self.render_mode = render_mode
        self.clip1 = clip1
        self.clip2 = clip2
        # mutation
        self.mutation_enabled = bool(mutation)
        self.mutation_sz = int(mutation_patch_size)
        self.mutation_p = float(mutation_prob)
        # food
        self.food_enabled = bool(food_enabled)
        self.food_spawn_interval = int(food_spawn_interval)
        self.food_n_patches = int(food_n_patches)
        self.food_patch_size = int(food_patch_size)
        self.food_amount = float(food_amount)
        self.food_consume_rate = float(food_consume_rate)
        self.food_bonus = float(food_bonus)
        self.mass_decay = float(mass_decay)
        self.food_green_channel = int(food_green_channel)
        self.food_auto_size = bool(food_auto_size)
        self.food_conv_mode = food_conv_mode
        self.food_vis_scale = float(food_vis_scale)
        self.food_vis_color = tuple(food_vis_color)
        self.food_diffusion_alpha = float(food_diffusion_alpha)

        # Connectivity: by default, all k kernels read from channel 0 and
        # contribute to channel 0 (for C=1). For C>1, still route all to ch 0.
        c0, c1 = conn_from_matrix(M)
        # Use connectivity-implied number of kernels
        self.k = len(c0)
        self.cfg = Config(
            X=grid_size,
            Y=grid_size,
            C=C,
            c0=c0,
            c1=c1,
            k=self.k,
            dd=dd,
            dt=dt,
            sigma=sigma,
            border=border,
            mix_rule=mix_rule,
        )
        # Parameter ranges from FlowLeniaParams initializations
        self.bounds = {
            "R": (2.0, 25.0),
            "r": (0.2, 1.0),
            "m": (0.05, 0.5),
            "s": (0.001, 0.18),
            "a": (0.0, 1.0),
            "b": (0.001, 1.0),
            "w": (0.01, 0.5),
            "fcr": (0.0, 0.5),
        }

        # Sample a deterministic base set of parameters following the same ranges
        kR, kr, km, ks, ka, kb, kw = jr.split(jr.key(0), 7)
        base_R = jr.uniform(kR, (), minval=2.0, maxval=25.0)
        base_r = jr.uniform(kr, (self.k,), minval=0.2, maxval=1.0)
        base_m = jr.uniform(km, (self.k,), minval=0.05, maxval=0.5)
        base_s = jr.uniform(ks, (self.k,), minval=0.001, maxval=0.18)
        base_a = jr.uniform(ka, (self.k, 3), minval=0.0, maxval=1.0)
        base_b = jr.uniform(kb, (self.k, 3), minval=0.001, maxval=1.0)
        base_w = jr.uniform(kw, (self.k, 3), minval=0.01, maxval=0.5)

        # Flattened base in logit space around which we take deltas (Lenia-style)
        def to_raw(x, name):
            lo, hi = self.bounds[name]
            norm = jnp.clip((x - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            return inv_sigmoid(norm)

        raw_R = to_raw(base_R, "R").reshape((-1,))
        raw_r = to_raw(base_r, "r").reshape((-1,))
        raw_m = to_raw(base_m, "m").reshape((-1,))
        raw_s = to_raw(base_s, "s").reshape((-1,))
        raw_a = to_raw(base_a, "a").reshape((-1,))
        raw_b = to_raw(base_b, "b").reshape((-1,))
        raw_w = to_raw(base_w, "w").reshape((-1,))
        # Base food consume rate uses current hyper as center
        base_fcr = jnp.clip(jnp.array(self.food_consume_rate), self.bounds["fcr"][0], self.bounds["fcr"][1])
        raw_fcr = inv_sigmoid(jnp.clip((base_fcr - self.bounds["fcr"][0])/(self.bounds["fcr"][1]-self.bounds["fcr"][0]), 1e-6, 1-1e-6)).reshape((-1,))

        self.base_dyn_raw = jnp.concatenate([raw_R, raw_r, raw_m, raw_s, raw_a, raw_b, raw_w, raw_fcr], axis=0)

        # Precompute flat bounds for denormalization during init_state
        self._dyn_lo = jnp.concatenate([
            jnp.full((1,), self.bounds["R"][0]),
            jnp.full((self.k,), self.bounds["r"][0]),
            jnp.full((self.k,), self.bounds["m"][0]),
            jnp.full((self.k,), self.bounds["s"][0]),
            jnp.full((self.k * 3,), self.bounds["a"][0]),
            jnp.full((self.k * 3,), self.bounds["b"][0]),
            jnp.full((self.k * 3,), self.bounds["w"][0]),
            jnp.full((1,), self.bounds["fcr"][0]),
        ], axis=0)
        self._dyn_hi = jnp.concatenate([
            jnp.full((1,), self.bounds["R"][1]),
            jnp.full((self.k,), self.bounds["r"][1]),
            jnp.full((self.k,), self.bounds["m"][1]),
            jnp.full((self.k,), self.bounds["s"][1]),
            jnp.full((self.k * 3,), self.bounds["a"][1]),
            jnp.full((self.k * 3,), self.bounds["b"][1]),
            jnp.full((self.k * 3,), self.bounds["w"][1]),
            jnp.full((1,), self.bounds["fcr"][1]),
        ], axis=0)

    # ---------- params interface ----------
    def default_params(self, rng):
        # Single flat vector of deltas like Lenia (dynamics only)
        n_dyn = self.base_dyn_raw.size
        return jr.normal(rng, (n_dyn,)) * 0.1

    # ---------- state interface ----------
    def init_state(self, rng, params):
        # Apply deltas to dynamics only
        n_dyn = self.base_dyn_raw.size
        dyn_delta = params[:n_dyn]
        raw_dyn = self.base_dyn_raw + jnp.clip(dyn_delta, -self.clip1, self.clip1)
        norm_dyn = jax.nn.sigmoid(raw_dyn)

        # Denormalize and reshape to parameter tensors
        dyn_vals = self._dyn_lo + norm_dyn * (self._dyn_hi - self._dyn_lo)
        # Unflatten
        idx = 0
        R = dyn_vals[idx]; idx += 1
        r = dyn_vals[idx:idx + self.k]; idx += self.k
        m = dyn_vals[idx:idx + self.k]; idx += self.k
        s = dyn_vals[idx:idx + self.k]; idx += self.k
        a = dyn_vals[idx:idx + self.k * 3].reshape((self.k, 3)); idx += self.k * 3
        b = dyn_vals[idx:idx + self.k * 3].reshape((self.k, 3)); idx += self.k * 3
        w = dyn_vals[idx:idx + self.k * 3].reshape((self.k, 3)); idx += self.k * 3
        fcr = dyn_vals[idx]; idx += 1

        # Prebuild reintegration operator (kept on substrate, not in state)
        self.RT = ReintegrationTracking(
            SX=self.cfg.X, SY=self.cfg.Y, dt=self.cfg.dt, dd=self.cfg.dd,
            sigma=self.cfg.sigma, border=self.cfg.border, has_hidden=True,
            mix=self.cfg.mix_rule,
        )
        # Compute kernels fft once from parameters and start with zeros
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, R, r, a, w, b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.k))
        Food = jnp.zeros((self.cfg.X, self.cfg.Y))

        # Seed patch(es): notebook-style or random
        kA, kP = split(rng)
        sz = int(self.seed_patch_size)
        sz = max(1, min(sz, self.grid_size))
        mode = self.seed_mode
        if mode == "center":
            i0 = self.grid_size // 2 - sz // 2
            j0 = self.grid_size // 2 - sz // 2
            A_patch = jr.uniform(kA, (sz, sz, self.C))
            if self.p_constant_per_patch:
                P_vec = jr.uniform(kP, (1, 1, self.k))
                P_patch = jnp.ones((sz, sz, self.k)) * P_vec
            else:
                P_patch = jr.uniform(kP, (sz, sz, self.k))
            A = jax.lax.dynamic_update_slice(A, A_patch, (i0, j0, 0))
            P = jax.lax.dynamic_update_slice(P, P_patch, (i0, j0, 0))
        elif mode == "notebook_centers":
            # Five patches: four near corners and one center, scaled to grid
            g = self.grid_size
            max_i = g - sz
            max_j = g - sz
            # Use ratios similar to the notebook (~0.27, 0.73, 0.5)
            r1, r2, rC = 0.27, 0.73, 0.5
            centers = [
                (int(round(g * r1)), int(round(g * r1))),
                (int(round(g * r1)), int(round(g * r2))),
                (int(round(g * r2)), int(round(g * r1))),
                (int(round(g * r2)), int(round(g * r2))),
                (int(round(g * rC)), int(round(g * rC))),
            ]
            i0s = jnp.array([max(0, min(ci - sz // 2, max_i)) for ci, _ in centers], dtype=jnp.int32)
            j0s = jnp.array([max(0, min(cj - sz // 2, max_j)) for _, cj in centers], dtype=jnp.int32)

            def body(t, carry):
                A_cur, P_cur, key_cur = carry
                key_next, kA_t, kP_t = jr.split(key_cur, 3)
                i0 = i0s[t]
                j0 = j0s[t]
                A_patch = jr.uniform(kA_t, (sz, sz, self.C))
                if self.p_constant_per_patch:
                    P_vec = jr.uniform(kP_t, (1, 1, self.k))
                    P_patch = jnp.ones((sz, sz, self.k)) * P_vec
                else:
                    P_patch = jr.uniform(kP_t, (sz, sz, self.k))
                A_cur = jax.lax.dynamic_update_slice(A_cur, A_patch, (i0, j0, 0))
                P_cur = jax.lax.dynamic_update_slice(P_cur, P_patch, (i0, j0, 0))
                return (A_cur, P_cur, key_next)

            A, P, _ = jax.lax.fori_loop(0, 5, body, (A, P, kA))
        else:
            # 'random_patches': place N random patches, overlaps allowed
            n_target = max(1, int(self.seed_n_patches))
            max_i = self.grid_size - sz
            max_j = self.grid_size - sz
            key_loop = kA

            def body(t, carry):
                A_cur, P_cur, key_cur = carry
                # Split keys for position and patch
                key_next, kA_t, kP_t, ki, kj = jr.split(key_cur, 5)
                i0 = jr.randint(ki, (), 0, max_i + 1)
                j0 = jr.randint(kj, (), 0, max_j + 1)
                A_patch = jr.uniform(kA_t, (sz, sz, self.C))
                if self.p_constant_per_patch:
                    P_vec = jr.uniform(kP_t, (1, 1, self.k))
                    P_patch = jnp.ones((sz, sz, self.k)) * P_vec
                else:
                    P_patch = jr.uniform(kP_t, (sz, sz, self.k))
                A_cur = jax.lax.dynamic_update_slice(A_cur, A_patch, (i0, j0, 0))
                P_cur = jax.lax.dynamic_update_slice(P_cur, P_patch, (i0, j0, 0))
                return (A_cur, P_cur, key_next)

            A, P, _ = jax.lax.fori_loop(0, n_target, body, (A, P, key_loop))

        # step counter for scheduled events (food spawn)
        t = jnp.array(0, dtype=jnp.int32)
        state = {"A": A, "P": P, "fK": fK, "m": m, "s": s, "fcr": fcr, "Food": Food, "t": t}
        # Step once to avoid trivial zero image, like Lenia
        return self.step_state(rng, state, params)

    def step_state(self, rng, state, params):
        # params are unused (Lenia-style), dynamics are embedded in state
        A, P, fK = state["A"], state["P"], state["fK"]
        m, s = state["m"], state["s"]
        Food = state.get("Food", jnp.zeros(A.shape[:2]))
        t = state.get("t", jnp.array(0, dtype=jnp.int32))
        fcr = state.get("fcr", jnp.array(self.food_consume_rate))

        # Convolution in Fourier domain per FlowLenia
        fA = jnp.fft.fft2(A, axes=(0, 1))
        fAk = fA[:, :, self.cfg.c0]
        U = jnp.real(jnp.fft.ifft2(fK * fAk, axes=(0, 1)))
        U = growth(U, m, s) * P
        U = jnp.dstack([U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C)])

        # Flow field and reintegration
        F = sobel(U)
        C_grad = sobel(A.sum(axis=-1, keepdims=True))
        alpha = jnp.clip((A[:, :, None, :] / 2) ** 2, 0.0, 1.0)
        mag = self.cfg.dd - self.cfg.sigma
        F = jnp.clip(F * (1 - alpha) - C_grad * alpha, -mag, mag)

        nA, nP = self.RT(A, P, F)

        # Optional mutation: inject a random parameter patch into P
        if self.mutation_enabled:
            kmut, kpos, kprob = jr.split(rng, 3)
            sz = max(1, min(self.mutation_sz, nP.shape[0], nP.shape[1]))
            kdim = nP.shape[-1]
            # mutation tensor and location
            mut = jnp.ones((sz, sz, kdim)) * jr.normal(kmut, (1, 1, kdim))
            max_i = nP.shape[0] - sz
            max_j = nP.shape[1] - sz
            ki, kj = jr.split(kpos)
            i0 = jr.randint(ki, (), 0, max_i + 1)
            j0 = jr.randint(kj, (), 0, max_j + 1)
            dP = jax.lax.dynamic_update_slice(jnp.zeros_like(nP), mut, (i0, j0, 0))
            msk = (jr.uniform(kprob, ()) < self.mutation_p).astype(nP.dtype)
            nP = nP + dP * msk

        # Optional mass decay and food mechanics
        if self.food_enabled or (self.mass_decay > 0):
            # global decay on A
            if self.mass_decay > 0:
                nA = nA * (1.0 - self.mass_decay)

            if self.food_enabled:
                # Track mass at cycle start to compensate observed loss (measured, not predicted)
                mass_cycle_start = state.get("mass_cycle_start", jnp.sum(nA))
                mass_cur = jnp.sum(nA)
                # periodic spawn; also force a spawn on the very first step to seed food immediately
                first_spawn = (t == 0)
                periodic_spawn = (self.food_spawn_interval > 0) & (jnp.mod(t, self.food_spawn_interval) == 0)
                do_spawn = first_spawn | periodic_spawn
                def spawn_food(Food_in, key, required_food):
                    # Use max_sz as static patch container; fp_sz is fixed by food_sz (clipped)
                    max_sz = min(self.food_patch_size, Food_in.shape[0], Food_in.shape[1])
                    max_i = Food_in.shape[0] - max_sz
                    max_j = Food_in.shape[1] - max_sz
                    # Patch side is fixed by food_sz (clipped to grid); we auto-compute density if enabled
                    fp_sz = jnp.array(self.food_patch_size, dtype=jnp.int32)
                    fp_sz = jnp.clip(fp_sz, 0, max_sz)
                    n = int(self.food_n_patches)
                    cells_per_patch = jnp.maximum(1.0, jnp.float32(fp_sz) * jnp.float32(fp_sz))
                    patches = float(max(1, n))
                    # distribute required food evenly across the fixed patch area
                    # when auto_size is on, enforce a per-cell floor of food_amount so we never stop spawning completely
                    auto_density = required_food / (cells_per_patch * patches + 1e-8)
                    food_amt_cell = jnp.where(
                        self.food_auto_size,
                        jnp.maximum(auto_density, jnp.array(self.food_amount, dtype=jnp.float32)),
                        jnp.array(self.food_amount, dtype=jnp.float32),
                    )
                    def body(i, carry):
                        Fcur, kcur = carry
                        kcur, ki, kj = jr.split(kcur, 3)
                        i0 = jr.randint(ki, (), 0, max_i + 1)
                        j0 = jr.randint(kj, (), 0, max_j + 1)
                        rows = jnp.arange(max_sz) < fp_sz
                        mask = (rows[:, None] & rows[None, :]).astype(jnp.float32)
                        patch = mask * food_amt_cell
                        Fcur = Fcur + jax.lax.dynamic_update_slice(jnp.zeros_like(Fcur), patch, (i0, j0))
                        return (Fcur, kcur)
                    Food_add, _ = jax.lax.fori_loop(0, n, body, (jnp.zeros_like(Food_in), rng))
                    return Food_in + Food_add
                # Observed loss since last cycle start
                observed_loss = jnp.maximum(0.0, mass_cycle_start - mass_cur)
                required_food = jnp.where(self.food_auto_size, observed_loss / (self.food_bonus + 1e-8), 0.0)
                Food = jax.lax.select(do_spawn, spawn_food(Food, rng, required_food), Food)
                # Reset cycle start mass after this step to the actual post-update mass
                # so next cycle measures true observed loss (not an inflated target).
                mass_cycle_start = mass_cycle_start  # will be overwritten below if we spawned

                # consumption: only green channel consumes
                gc = int(self.food_green_channel)
                gc = max(0, min(gc, int(nA.shape[-1]) - 1))
                A_g = nA[..., gc]
                # amount to consume per pixel this step
                if self.food_conv_mode == 'conv':
                    # stationary food, identity kernel => signal equals Food
                    rate_field = fcr * Food
                else:
                    rate_field = fcr * A_g
                eat = jnp.minimum(Food, rate_field)
                Food = Food - eat
                # convert to mass in green channel with bonus
                nA = nA.at[..., gc].add(eat * self.food_bonus)
                # After consumption, record true cycle start mass when we spawned this step
                new_mass_cycle_start = jnp.sum(nA)
                mass_cycle_start = jax.lax.select(do_spawn & self.food_auto_size, new_mass_cycle_start, mass_cycle_start)

                # Optional food diffusion (conservative blur)
                if self.food_diffusion_alpha > 0:
                    kernel = jnp.array([[1.0, 2.0, 1.0],
                                        [2.0, 4.0, 2.0],
                                        [1.0, 2.0, 1.0]], dtype=Food.dtype) / 16.0
                    boundary = 'wrap' if self.border == 'torus' else 'symm'
                    diffused = jsp.signal.convolve2d(Food, kernel, mode='same', boundary=boundary)
                    a = self.food_diffusion_alpha
                    Food = (1.0 - a) * Food + a * diffused

            t = t + jnp.array(1, dtype=jnp.int32)

        return {"A": nA, "P": nP, "fK": fK, "m": m, "s": s, "Food": Food, "t": t, "mass_cycle_start": mass_cycle_start}

    def render_state(self, state, params, img_size=None):
        mode = getattr(self, 'render_mode', 'A')
        A = state["A"]
        if mode == 'Pcolor':
            # Notebook-style: intensity = sum(A) times first 3 channels of P
            P = state["P"]
            inten = A.sum(axis=-1, keepdims=True)
            # Use first three P channels for RGB; clip to [0,1]
            img = jnp.clip(inten * P[..., :3], 0.0, 1.0)
        else:
            # Render activations A directly
            C = A.shape[-1]
            if C == 1:
                img = jnp.dstack([A[..., 0], A[..., 0], A[..., 0]])
            elif C == 2:
                img = jnp.dstack([A[..., 0], A[..., 0], A[..., 1]])
            else:
                img = A[..., :3]
        # overlay food (brown)
        Food = state.get('Food', None)
        if Food is not None:
            f = Food * self.food_vis_scale
            f = f / (1.0 + f)
            overlay_color = jnp.array(self.food_vis_color, dtype=img.dtype)
            overlay = f[..., None] * overlay_color[None, None, :]
            img = jnp.clip(img + overlay, 0.0, 1.0)
        
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method="nearest")
        return img
