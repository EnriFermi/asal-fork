#!/usr/bin/env python3
"""
Block-Transformer meta-optimizer trained with PPO on MNIST/CIFAR-10.

Core idea:
- Base CNN is re-initialized each RL episode.
- Each inner optimization step:
  * compute gradients on a minibatch
  * build per-parameter features [w, g, m]
    where w is weight value, g is gradient, m is momentum EMA of g
  * split the flattened parameter vector into small blocks
  * run a tiny Transformer over each block-as-sequence (batched with padding)
  * output a bounded multiplicative correction of an SGD-like step
  * sample delta from factorized Normal(mean, std), apply to ALL parameters
- Reward targets:
  * val_acc: terminal validation accuracy after inner steps (sparse)
  * train_loss: -train loss at each step (dense)
  * weighted_train_loss: -w_t * train loss with early-step emphasis (dense)
  * mixed: dense -train loss + terminal validation accuracy bonus
- Train the update rule with PPO.

Dependencies: torch, torchvision
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def make_datasets(name: str, data_root: str):
    name = name.lower()
    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        return train, test

    if name in {"cifar", "cifar10"}:
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm_test)
        return train, test

    raise ValueError(f"Unknown dataset: {name}")


def sample_subset(dataset, n, rng: random.Random):
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    return Subset(dataset, idx[:n])


class SmallMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallCIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_base_model(dataset_name: str):
    if dataset_name.lower() == "mnist":
        return SmallMNISTCNN()
    return SmallCIFARCNN()


def get_params(model):
    return [p for p in model.parameters() if p.requires_grad]


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


class BlockTransformerPolicy(nn.Module):
    """
    Block-wise transformer producing per-parameter update means.

    Safe parameterization:
        r = transformer([w, g, m])
        corr = 1 + resid_scale * tanh(r)
        mean_delta = -lr * corr * g

    Exploration noise is gradient-scaled:
        std = exp(clamped_log_std) * lr * (|g| + eps)

    lr is positive and capped by max_lr.
    """

    def __init__(self, d_model=32, nhead=4, num_layers=1,
                 block_size=128, init_sgd_lr=0.05, resid_scale=0.1,
                 max_lr=0.05, log_std_init=-8.0, log_std_min=-12.0, log_std_max=-4.0):
        super().__init__()
        self.block_size = block_size
        self.resid_scale = float(resid_scale)

        self.max_lr = float(max_lr)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.embed = nn.Linear(3, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

        self.base_lr = nn.Parameter(torch.tensor(float(init_sgd_lr)))
        self.log_std = nn.Parameter(torch.tensor(float(log_std_init)))

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _lr(self):
        return torch.clamp(F.softplus(self.base_lr), max=self.max_lr)

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

    def residual_vec(self, w_flat, g_flat, m_flat):
        w_b, g_b, m_b, mask, N = self._make_blocks(w_flat, g_flat, m_flat)
        tokens = torch.stack([w_b, g_b, m_b], dim=-1)  # (nb, B, 3)
        x = self.embed(tokens)
        y = self.encoder(x, src_key_padding_mask=mask)
        r = self.head(y).squeeze(-1)                   # (nb, B)
        return r.reshape(-1)[:N]

    def mean_delta(self, w_flat, g_flat, m_flat):
        r = self.residual_vec(w_flat, g_flat, m_flat)
        corr = 1.0 + self.resid_scale * torch.tanh(r)
        return -self._lr() * corr * g_flat

    def dist(self, w_flat, g_flat, m_flat):
        mean = self.mean_delta(w_flat, g_flat, m_flat)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        base = self._lr() * (g_flat.abs() + 1e-8)
        std = torch.exp(log_std) * base
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


class ValueNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def value_features(w_flat, g_flat, m_flat):
    mean_w = w_flat.mean()
    std_w = w_flat.std(unbiased=False)
    mean_g = g_flat.mean()
    std_g = g_flat.std(unbiased=False)
    mean_abs_g = g_flat.abs().mean()
    mean_m = m_flat.mean()
    std_m = m_flat.std(unbiased=False)
    return torch.stack([mean_w, std_w, mean_g, std_g, mean_abs_g, mean_m, std_m])


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

    def add(self, w_flat, g_flat, m_flat, vfeat, action, logprob, value, reward, done):
        self.w.append(w_flat.detach().cpu())
        self.g.append(g_flat.detach().cpu())
        self.m.append(m_flat.detach().cpu())
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


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    accs = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        accs.append(accuracy_from_logits(logits, y))
    model.train()
    return float(sum(accs) / max(1, len(accs)))


def run_episode(policy, value_net, train_ds, test_ds, args, device, rng):
    model = build_base_model(args.dataset).to(device)
    model.train()

    train_sub = sample_subset(train_ds, args.train_size, rng)
    val_sub = sample_subset(test_ds, args.val_size, rng)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size,
                            shuffle=False)

    params = get_params(model)
    N = sum(p.numel() for p in params)
    m_flat = torch.zeros(N, device=device)

    data_iter = iter(train_loader)
    buf = Buffer()

    weights = None
    if args.reward_mode == "weighted_train_loss":
        w = torch.tensor([args.loss_weight_alpha ** t for t in range(args.inner_steps)],
                         device=device, dtype=torch.float32)
        w = w / w.sum()
        weights = w.tolist()

    for step in range(args.inner_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        model.zero_grad(set_to_none=True)
        loss.backward()

        w_flat = flatten_params(params)
        g_flat = flatten_grads(params)

        m_flat = args.mom_beta * m_flat + (1.0 - args.mom_beta) * g_flat

        vfeat = value_features(w_flat, g_flat, m_flat)

        with torch.no_grad():
            action, logprob, _ = policy.act(w_flat, g_flat, m_flat)
            value = value_net(vfeat)

        with torch.no_grad():
            apply_update(params, action)

        loss_val = float(loss.detach().item())

        reward = 0.0
        done = False

        if args.reward_mode == "val_acc":
            if step == args.inner_steps - 1:
                acc = eval_accuracy(model, val_loader, device)
                reward = acc
                done = True

        elif args.reward_mode == "train_loss":
            reward = -args.step_loss_scale * loss_val
            if step == args.inner_steps - 1:
                done = True

        elif args.reward_mode == "weighted_train_loss":
            reward = -args.step_loss_scale * weights[step] * loss_val
            if step == args.inner_steps - 1:
                done = True

        elif args.reward_mode == "mixed":
            reward = -args.step_loss_scale * loss_val
            if step == args.inner_steps - 1:
                acc = eval_accuracy(model, val_loader, device)
                reward += args.final_val_scale * acc
                done = True

        buf.add(w_flat, g_flat, m_flat, vfeat, action,
                logprob.item(), value.item(), reward, done)

    return buf


def compute_gae(reward, value, done, gamma=0.99, lam=0.95):
    T = reward.size(0)
    adv = torch.zeros(T, device=reward.device)
    last = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        next_value = value[t + 1] if t + 1 < T else 0.0
        delta = reward[t] + gamma * next_value * mask - value[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + value
    return adv, ret


def ppo_update(policy, value_net, optimizer, buffer, args, device):
    w, g, m, vfeat, action, old_logprob, old_value, reward, done = buffer.stack(device)

    adv, ret = compute_gae(reward, old_value, done,
                           gamma=args.gamma, lam=args.lam)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    n = len(buffer)
    idx_all = torch.arange(n, device=device)

    for _ in range(args.ppo_epochs):
        perm = idx_all[torch.randperm(n)]
        for start in range(0, n, args.minibatch):
            mb = perm[start:start + args.minibatch]

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
            surr2 = torch.clamp(ratio, 1.0 - args.clip,
                                1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_value, ret[mb])
            entropy_bonus = entropy.mean()

            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) +
                                     list(value_net.parameters()),
                                     args.max_grad_norm)
            optimizer.step()


def run_baseline(train_ds, test_ds, args, device, rng, optim_name="adam"):
    model = build_base_model(args.dataset).to(device)
    params = get_params(model)

    train_sub = sample_subset(train_ds, args.train_size, rng)
    val_sub = sample_subset(test_ds, args.val_size, rng)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size,
                            shuffle=False)

    if optim_name == "adam":
        opt = torch.optim.Adam(params, lr=args.baseline_lr)
    elif optim_name == "sgd":
        opt = torch.optim.SGD(params, lr=args.baseline_lr, momentum=0.0)
    else:
        raise ValueError(optim_name)

    model.train()
    data_iter = iter(train_loader)

    for _ in range(args.inner_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return eval_accuracy(model, val_loader, device)


def episode_scalar(ep: Buffer, reward_mode: str):
    r = torch.stack(ep.reward)
    if reward_mode == "val_acc":
        return float(r[-1].item())
    return float(r.sum().item())


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "cifar10", "cifar"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--inner-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=256)

    p.add_argument("--mom-beta", type=float, default=0.9)

    p.add_argument("--ppo-updates", type=int, default=500)
    p.add_argument("--episodes-per-batch", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=16)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--reward-mode", type=str, default="weighted_train_loss",
                   choices=["val_acc", "train_loss", "weighted_train_loss", "mixed"])
    p.add_argument("--loss-weight-alpha", type=float, default=0.97)
    p.add_argument("--step-loss-scale", type=float, default=1.0)
    p.add_argument("--final-val-scale", type=float, default=0.0)

    p.add_argument("--resid-scale", type=float, default=0.1)
    p.add_argument("--max-lr", type=float, default=0.05)
    p.add_argument("--log-std-init", type=float, default=-8.0)
    p.add_argument("--log-std-min", type=float, default=-12.0)
    p.add_argument("--log-std-max", type=float, default=-4.0)

    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=1)

    p.add_argument("--outer-lr", type=float, default=3e-4)
    p.add_argument("--init-sgd-lr", type=float, default=0.05)

    p.add_argument("--baseline-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--no-baselines", action="store_true")

    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    train_ds, test_ds = make_datasets(args.dataset, args.data_root)

    policy = BlockTransformerPolicy(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        block_size=args.block_size,
        init_sgd_lr=args.init_sgd_lr,
        resid_scale=args.resid_scale,
        max_lr=args.max_lr,
        log_std_init=args.log_std_init,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    ).to(device)

    value_net = ValueNet(hidden=64).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.outer_lr
    )

    print(f"Dataset={args.dataset} device={device} inner_steps={args.inner_steps} "
          f"block_size={args.block_size} d_model={args.d_model} "
          f"reward_mode={args.reward_mode}")

    if not args.no_baselines and args.reward_mode == "val_acc":
        acc_adam = run_baseline(train_ds, test_ds, args, device, rng, "adam")
        acc_sgd = run_baseline(train_ds, test_ds, args, device, rng, "sgd")
        print(f"Baseline snapshot: Adam acc={acc_adam:.3f} | SGD acc={acc_sgd:.3f}")

    start_time = time.time()

    for upd in range(1, args.ppo_updates + 1):
        buffer = Buffer()
        ep_scores = []

        for _ in range(args.episodes_per_batch):
            ep = run_episode(policy, value_net, train_ds, test_ds, args, device, rng)
            score = episode_scalar(ep, args.reward_mode)
            ep_scores.append(score)

            buffer.w.extend(ep.w)
            buffer.g.extend(ep.g)
            buffer.m.extend(ep.m)
            buffer.vfeat.extend(ep.vfeat)
            buffer.action.extend(ep.action)
            buffer.logprob.extend(ep.logprob)
            buffer.value.extend(ep.value)
            buffer.reward.extend(ep.reward)
            buffer.done.extend(ep.done)

        mu = sum(ep_scores) / len(ep_scores)
        sd = (sum((r - mu) ** 2 for r in ep_scores) / len(ep_scores)) ** 0.5

        label = "batch mean final acc" if args.reward_mode == "val_acc" else "batch mean return"
        print(f"upd {upd:4d} | {label}={mu:.4f} std={sd:.4f}")

        ppo_update(policy, value_net, optimizer, buffer, args, device)

        if upd % args.eval_every == 0 or upd == 1:
            test_buf = run_episode(policy, value_net, train_ds, test_ds,
                                   args, device, rng)
            eval_score = episode_scalar(test_buf, args.reward_mode)

            lr_eff = policy._lr().item()
            std_val = torch.exp(torch.clamp(policy.log_std,
                                            policy.log_std_min,
                                            policy.log_std_max)).item()
            elapsed = time.time() - start_time

            if args.reward_mode == "val_acc":
                print(f"upd {upd:4d} | learned acc={eval_score:.3f} "
                      f"| lr_eff={lr_eff:.5f} | std_factor={std_val:.6f} | {elapsed:.1f}s")
                if not args.no_baselines:
                    acc_adam = run_baseline(train_ds, test_ds, args, device, rng, "adam")
                    print(f"           | Adam acc={acc_adam:.3f}")
            else:
                print(f"upd {upd:4d} | eval return={eval_score:.4f} "
                      f"| lr_eff={lr_eff:.5f} | std_factor={std_val:.6f} | {elapsed:.1f}s")

    print("Done.")


if __name__ == "__main__":
    main()
