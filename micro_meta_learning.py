#!/usr/bin/env python3
# Toy PPO meta-optimizer test with Adam baseline:
# - 2-layer MLP learns a simple 1D regression function.
# - Tiny block-transformer outputs absolute parameter updates.
# - Transformer is randomly initialized (NOT SGD-like).
# - Reward is negative validation MSE after a few inner steps.
# - Before PPO, we run an Adam baseline over many fresh tasks.

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Hardcoded experiment config
# ----------------------------
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task: y = sin(x) + 0.1 * x
N_TRAIN = 32
N_VAL = 32
X_MIN, X_MAX = -math.pi, math.pi

# Base model
HIDDEN = 32

# Inner optimization (episode length)
INNER_STEPS = 8
MOM_BETA = 0.9

# Meta-optimizer policy (tiny transformer)
BLOCK_SIZE = 32
D_MODEL = 32
NHEAD = 8
NUM_LAYERS = 3

# Update scaling to keep random policy safe
DELTA_SCALE = 0.02

# Exploration
LOG_STD_INIT = -4.0
LOG_STD_MIN = -10.0
LOG_STD_MAX = -3.0

# PPO
PPO_UPDATES = 5000
EPISODES_PER_BATCH = 64
PPO_EPOCHS = 1
MINIBATCH = 64

GAMMA = 1.0
LAMBDA = 1.0
CLIP = 0.1
ENT_COEF = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
OUTER_LR = 1e-4


PRINT_EVERY = 10

# Adam baseline
BASELINE_EPISODES = 128
ADAM_LR = 1e-3


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_xy(n):
    x = (X_MAX - X_MIN) * torch.rand(n, 1, device=DEVICE) + X_MIN
    y = torch.sin(x) + 0.1 * x
    return x, y


def flatten_params(params):
    return torch.cat([p.detach().view(-1) for p in params])


def flatten_grads(params):
    gs = []
    for p in params:
        if p.grad is None:
            gs.append(torch.zeros_like(p).view(-1))
        else:
            gs.append(p.grad.detach().view(-1))
    return torch.cat(gs)


@torch.no_grad()
def apply_update(params, delta_flat):
    i = 0
    for p in params:
        n = p.numel()
        p.add_(delta_flat[i:i + n].view_as(p))
        i += n


def value_features(w_flat, g_flat, m_flat):
    mean_w = w_flat.mean()
    std_w = w_flat.std(unbiased=False)
    mean_g = g_flat.mean()
    std_g = g_flat.std(unbiased=False)
    mean_abs_g = g_flat.abs().mean()
    mean_m = m_flat.mean()
    std_m = m_flat.std(unbiased=False)
    return torch.stack([mean_w, std_w, mean_g, std_g, mean_abs_g, mean_m, std_m])


# ----------------------------
# Base model (2-layer MLP)
# ----------------------------
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def get_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# ----------------------------
# Policy: small block transformer
# ----------------------------
class BlockTransformerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = BLOCK_SIZE

        self.embed = nn.Linear(3, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=4 * D_MODEL,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(D_MODEL, 1)

        self.log_std = nn.Parameter(torch.tensor(LOG_STD_INIT))

    def _make_blocks(self, w, g, m):
        N = w.numel()
        B = self.block_size
        nb = (N + B - 1) // B
        pad = nb * B - N

        if pad > 0:
            z = torch.zeros(pad, device=w.device, dtype=w.dtype)
            w = torch.cat([w, z])
            g = torch.cat([g, z])
            m = torch.cat([m, z])

        w_b = w.view(nb, B)
        g_b = g.view(nb, B)
        m_b = m.view(nb, B)

        mask = torch.zeros(nb * B, device=w.device, dtype=torch.bool)
        if pad > 0:
            mask[-pad:] = True
        mask = mask.view(nb, B)

        return w_b, g_b, m_b, mask, N

    def mean_delta(self, w_flat, g_flat, m_flat):
        w_b, g_b, m_b, mask, N = self._make_blocks(w_flat, g_flat, m_flat)
        tokens = torch.stack([w_b, g_b, m_b], dim=-1)
        x = self.embed(tokens)
        y = self.encoder(x, src_key_padding_mask=mask)
        h = self.head(y).squeeze(-1).reshape(-1)[:N]
        return DELTA_SCALE * torch.tanh(h)

    def dist(self, w_flat, g_flat, m_flat):
        mean = self.mean_delta(w_flat, g_flat, m_flat)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) * torch.ones_like(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, w_flat, g_flat, m_flat):
        d = self.dist(w_flat, g_flat, m_flat)
        a = d.sample()
        logp = d.log_prob(a).sum()
        ent = d.entropy().sum()
        return a, logp, ent

    def logprob_of(self, w_flat, g_flat, m_flat, action):
        d = self.dist(w_flat, g_flat, m_flat)
        return d.log_prob(action).sum(), d.entropy().sum()


# ----------------------------
# Value function
# ----------------------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------
# PPO buffer
# ----------------------------
class Buffer:
    def __init__(self):
        self.w = []
        self.g = []
        self.m = []
        self.vfeat = []
        self.action = []
        self.logprob = []
        self.value = []
        self.reward = []
        self.done = []

    def add(self, w, g, m, vfeat, action, logprob, value, reward, done):
        self.w.append(w.detach().cpu())
        self.g.append(g.detach().cpu())
        self.m.append(m.detach().cpu())
        self.vfeat.append(vfeat.detach().cpu())
        self.action.append(action.detach().cpu())
        self.logprob.append(torch.tensor(logprob).cpu())
        self.value.append(torch.tensor(value).cpu())
        self.reward.append(torch.tensor(reward).cpu())
        self.done.append(torch.tensor(done).cpu())

    def stack(self, device):
        w = torch.stack(self.w).to(device)
        g = torch.stack(self.g).to(device)
        m = torch.stack(self.m).to(device)
        vfeat = torch.stack(self.vfeat).to(device)
        action = torch.stack(self.action).to(device)
        logprob = torch.stack(self.logprob).to(device)
        value = torch.stack(self.value).to(device)
        reward = torch.stack(self.reward).to(device)
        done = torch.stack(self.done).to(device).float()
        return w, g, m, vfeat, action, logprob, value, reward, done

    def __len__(self):
        return len(self.reward)


# ----------------------------
# Episode (policy-driven)
# ----------------------------
def run_episode(policy, value_net):
    model = TwoLayerMLP().to(DEVICE)
    params = get_params(model)
    N = sum(p.numel() for p in params)
    m_flat = torch.zeros(N, device=DEVICE)

    x_train, y_train = sample_xy(N_TRAIN)
    x_val, y_val = sample_xy(N_VAL)

    buf = Buffer()

    for step in range(INNER_STEPS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)

        model.zero_grad(set_to_none=True)
        loss.backward()

        w_flat = flatten_params(params)
        g_flat = flatten_grads(params)
        m_flat = MOM_BETA * m_flat + (1.0 - MOM_BETA) * g_flat

        vfeat = value_features(w_flat, g_flat, m_flat)

        with torch.no_grad():
            action, logprob, _ = policy.act(w_flat, g_flat, m_flat)
            value = value_net(vfeat)
            apply_update(params, action)

        reward = 0.0
        done = False
        if step == INNER_STEPS - 1:
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = F.mse_loss(val_pred, y_val).item()
            reward = -val_loss
            done = True

        buf.add(w_flat, g_flat, m_flat, vfeat, action,
                logprob.item(), value.item(), reward, done)

    return buf

# Adam baseline
BASELINE_EPISODES = 128
ADAM_LR = 1e-2
SGD_LR = 5e-2


def adam_episode_return():
    model = TwoLayerMLP().to(DEVICE)
    params = get_params(model)
    opt = torch.optim.Adam(params, lr=ADAM_LR)

    x_train, y_train = sample_xy(N_TRAIN)
    x_val, y_val = sample_xy(N_VAL)

    model.train()
    for _ in range(INNER_STEPS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = F.mse_loss(val_pred, y_val).item()
    return -val_loss


def sgd_episode_return():
    model = TwoLayerMLP().to(DEVICE)
    params = get_params(model)
    opt = torch.optim.SGD(params, lr=SGD_LR)

    x_train, y_train = sample_xy(N_TRAIN)
    x_val, y_val = sample_xy(N_VAL)

    model.train()
    for _ in range(INNER_STEPS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = F.mse_loss(val_pred, y_val).item()
    return -val_loss


def _mean_std(vals):
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    return mu, sd


def run_baselines():
    adam_rets = [adam_episode_return() for _ in range(BASELINE_EPISODES)]
    sgd_rets = [sgd_episode_return() for _ in range(BASELINE_EPISODES)]

    mu_a, sd_a = _mean_std(adam_rets)
    mu_s, sd_s = _mean_std(sgd_rets)

    print(f"Adam baseline ({BASELINE_EPISODES} tasks): mean return={mu_a:.4f} std={sd_a:.4f}")
    print(f"SGD  baseline ({BASELINE_EPISODES} tasks): mean return={mu_s:.4f} std={sd_s:.4f}")
    print("(return = -val_mse)")


def run_adam_baseline():
    rets = [adam_episode_return() for _ in range(BASELINE_EPISODES)]
    mu = sum(rets) / len(rets)
    sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
    print(f"Adam baseline over {BASELINE_EPISODES} tasks: mean return={mu:.4f} std={sd:.4f} "
          f"(return = -val_mse)")

# Adam baseline
BASELINE_EPISODES = 128
ADAM_LR = 1e-2
SGD_LR = 5e-2


def adam_episode_return():
    model = TwoLayerMLP().to(DEVICE)
    params = get_params(model)
    opt = torch.optim.Adam(params, lr=ADAM_LR)

    x_train, y_train = sample_xy(N_TRAIN)
    x_val, y_val = sample_xy(N_VAL)

    model.train()
    for _ in range(INNER_STEPS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = F.mse_loss(val_pred, y_val).item()
    return -val_loss


def sgd_episode_return():
    model = TwoLayerMLP().to(DEVICE)
    params = get_params(model)
    opt = torch.optim.SGD(params, lr=SGD_LR)

    x_train, y_train = sample_xy(N_TRAIN)
    x_val, y_val = sample_xy(N_VAL)

    model.train()
    for _ in range(INNER_STEPS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = F.mse_loss(val_pred, y_val).item()
    return -val_loss


def _mean_std(vals):
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    return mu, sd


def run_baselines():
    adam_rets = [adam_episode_return() for _ in range(BASELINE_EPISODES)]
    sgd_rets = [sgd_episode_return() for _ in range(BASELINE_EPISODES)]

    mu_a, sd_a = _mean_std(adam_rets)
    mu_s, sd_s = _mean_std(sgd_rets)

    print(f"Adam baseline ({BASELINE_EPISODES} tasks): mean return={mu_a:.4f} std={sd_a:.4f}")
    print(f"SGD  baseline ({BASELINE_EPISODES} tasks): mean return={mu_s:.4f} std={sd_s:.4f}")
    print("(return = -val_mse)")

# ----------------------------
# GAE + PPO update
# ----------------------------
def compute_gae(reward, value, done):
    T = reward.size(0)
    adv = torch.zeros(T, device=reward.device)
    last = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        next_value = value[t + 1] if t + 1 < T else 0.0
        delta = reward[t] + GAMMA * next_value * mask - value[t]
        last = delta + GAMMA * LAMBDA * mask * last
        adv[t] = last
    ret = adv + value
    return adv, ret


def ppo_update(policy, value_net, optim, buffer):
    w, g, m, vfeat, action, old_logprob, old_value, reward, done = buffer.stack(DEVICE)

    adv, ret = compute_gae(reward, old_value, done)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    n = len(buffer)
    idx_all = torch.arange(n, device=DEVICE)

    for _ in range(PPO_EPOCHS):
        perm = idx_all[torch.randperm(n)]
        for start in range(0, n, MINIBATCH):
            mb = perm[start:start + MINIBATCH]

            new_logprob_list = []
            entropy_list = []
            for i in mb:
                lp, ent = policy.logprob_of(w[i], g[i], m[i], action[i])
                new_logprob_list.append(lp)
                entropy_list.append(ent)
            new_logprob = torch.stack(new_logprob_list)
            entropy = torch.stack(entropy_list)

            new_value = value_net(vfeat[mb])

            ratio = torch.exp(new_logprob - old_logprob[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_value, ret[mb])

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()),
                                     MAX_GRAD_NORM)
            optim.step()


# ----------------------------
# Main training loop
# ----------------------------
def main():
    set_seed(SEED)

    print(f"Device={DEVICE} | INNER_STEPS={INNER_STEPS} | "
          f"task: y=sin(x)+0.1x | reward=-val_mse")

    run_baselines()


    policy = BlockTransformerPolicy().to(DEVICE)
    value_net = ValueNet().to(DEVICE)

    optim = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()),
                             lr=OUTER_LR)

    t0 = time.time()

    for upd in range(1, PPO_UPDATES + 1):
        big = Buffer()
        ep_rewards = []

        for _ in range(EPISODES_PER_BATCH):
            ep = run_episode(policy, value_net)
            ep_r = float(torch.stack(ep.reward).sum().item())
            ep_rewards.append(ep_r)

            big.w.extend(ep.w)
            big.g.extend(ep.g)
            big.m.extend(ep.m)
            big.vfeat.extend(ep.vfeat)
            big.action.extend(ep.action)
            big.logprob.extend(ep.logprob)
            big.value.extend(ep.value)
            big.reward.extend(ep.reward)
            big.done.extend(ep.done)

        mu = sum(ep_rewards) / len(ep_rewards)
        sd = (sum((r - mu) ** 2 for r in ep_rewards) / len(ep_rewards)) ** 0.5

        ppo_update(policy, value_net, optim, big)

        if upd == 1 or upd % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            log_std = float(torch.clamp(policy.log_std, LOG_STD_MIN, LOG_STD_MAX).item())
            print(f"upd {upd:4d} | mean return={mu:.4f} std={sd:.4f} | log_std={log_std:.3f} | {elapsed:.1f}s")

    
    ep = run_episode(policy, value_net)
    final_r = float(torch.stack(ep.reward).sum().item())
    print(f"final probe return={final_r:.4f} (higher is better, equals -val_mse)")


if __name__ == "__main__":
    main()
