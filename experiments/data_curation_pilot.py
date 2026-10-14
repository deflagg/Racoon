from __future__ import annotations

import itertools
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raccoon import Raccoon

torch.set_num_threads(1)

FWD = [2, 3, 4, 5, 6, 7]
BWD = [2, 7, 6, 5, 4, 3]

SEEDS = list(range(6))
STEPS = 5000
BATCH_SIZE = 24
LR = 3e-3
SCHEDULE_SEED = 999
RANDOM_DATA_SEED = 20261013


def seq_from_cycle(mode: int, start: int, cycle: list[int]) -> list[int]:
    i = cycle.index(start)
    return [mode] + cycle[i:] + cycle[:i]


def target_for_mode(seq: list[int], mode: int) -> list[int]:
    cycle = FWD if mode == 0 else BWD
    return [-100] + [cycle[(cycle.index(tok) + 1) % 6] for tok in seq[1:]]


def baseline_data():
    xs, ys = [], []
    for mode, cycle in [(0, FWD), (1, BWD)]:
        for start in range(2, 8):
            seq = seq_from_cycle(mode, start, cycle)
            xs.append(seq)
            ys.append(target_for_mode(seq, mode))
    return torch.tensor(xs), torch.tensor(ys)


def conflict_data():
    xs, ymodes, ytrajs, meta = [], [], [], []
    for mode in (0, 1):
        opposite = BWD if mode == 0 else FWD
        for start in range(2, 8):
            seq = seq_from_cycle(mode, start, opposite)
            xs.append(seq)
            ymodes.append(target_for_mode(seq, mode))
            ytrajs.append(
                [-100]
                + [opposite[(opposite.index(tok) + 1) % 6] for tok in seq[1:]]
            )
            meta.append((mode, start))
    return (
        torch.tensor(xs),
        torch.tensor(ymodes),
        torch.tensor(ytrajs),
        meta,
    )


class ParallelRaccoon(nn.Module):
    """Stack independent Raccoon seeds into one batched parameter tensor.

    There is no communication or shared optimizer state between seeds.
    The summed per-seed loss gives each parameter slice the same gradient it
    would receive from its own seed's mean cross-entropy loss.
    """

    def __init__(self, seeds: list[int]):
        super().__init__()
        states = []
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            states.append(
                {k: v.detach().clone() for k, v in Raccoon().state_dict().items()}
            )

        def p(name: str) -> nn.Parameter:
            return nn.Parameter(torch.stack([state[name] for state in states], dim=0))

        self.token_emb = p("token_emb.weight")
        self.pos_emb = p("pos_emb.weight")

        self.ln1_w = p("block.ln1.weight")
        self.ln1_b = p("block.ln1.bias")
        self.q_w = p("block.attn.q_proj.weight")
        self.q_b = p("block.attn.q_proj.bias")
        self.k_w = p("block.attn.k_proj.weight")
        self.k_b = p("block.attn.k_proj.bias")
        self.v_w = p("block.attn.v_proj.weight")
        self.v_b = p("block.attn.v_proj.bias")
        self.o_w = p("block.attn.o_proj.weight")
        self.o_b = p("block.attn.o_proj.bias")

        self.ln2_w = p("block.ln2.weight")
        self.ln2_b = p("block.ln2.bias")
        self.fc1_w = p("block.fc1.weight")
        self.fc1_b = p("block.fc1.bias")
        self.fc2_w = p("block.fc2.weight")
        self.fc2_b = p("block.fc2.bias")

        self.lnf_w = p("ln_f.weight")
        self.lnf_b = p("ln_f.bias")

    @staticmethod
    def linear(x, w, b):
        return torch.einsum("sbti,soi->sbto", x, w) + b[:, None, None, :]

    @staticmethod
    def layer_norm(x, w, b, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (
            (x - mean) * torch.rsqrt(var + eps) * w[:, None, None, :]
            + b[:, None, None, :]
        )

    def forward(self, tokens):
        _, seq_len = tokens.shape

        x = self.token_emb[:, tokens, :] + self.pos_emb[:, :seq_len, :][:, None, :, :]

        n1 = self.layer_norm(x, self.ln1_w, self.ln1_b)
        q = self.linear(n1, self.q_w, self.q_b)
        k = self.linear(n1, self.k_w, self.k_b)
        v = self.linear(n1, self.v_w, self.v_b)

        scores = torch.einsum("sbtd,sbud->sbtu", q, k) / math.sqrt(4)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        y = torch.einsum("sbtu,sbud->sbtd", attn, v)
        x = x + self.linear(y, self.o_w, self.o_b)

        n2 = self.layer_norm(x, self.ln2_w, self.ln2_b)
        h = F.gelu(self.linear(n2, self.fc1_w, self.fc1_b))
        x = x + self.linear(h, self.fc2_w, self.fc2_b)

        x = self.layer_norm(x, self.lnf_w, self.lnf_b)
        return torch.einsum("sbtd,svd->sbtv", x, self.token_emb)


def per_seed_loss(logits, targets):
    logp = F.log_softmax(logits, dim=-1)
    mask = targets != -100
    gather_idx = (
        targets.clamp_min(0)[None, :, :, None]
        .expand(logits.shape[0], -1, -1, 1)
    )
    chosen = logp.gather(-1, gather_idx).squeeze(-1)
    return -(chosen * mask[None]).sum(dim=(1, 2)) / mask.sum()


def train_condition(x, y):
    model = ParallelRaccoon(SEEDS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SCHEDULE_SEED)
    schedule = torch.randint(0, x.shape[0], (STEPS, BATCH_SIZE), generator=generator)

    for step in range(STEPS):
        xb = x[schedule[step]]
        yb = y[schedule[step]]
        losses = per_seed_loss(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()

    return model


@torch.no_grad()
def evaluate(model, x_base, y_base, x_hold, y_mode, y_traj):
    base_pred = model(x_base).argmax(dim=-1)
    base_mask = y_base != -100
    base_acc = (
        ((base_pred == y_base[None]) & base_mask[None]).sum(dim=(1, 2))
        / base_mask.sum()
    )

    pred = model(x_hold).argmax(dim=-1)
    mask = y_mode != -100
    mode_acc = (
        ((pred == y_mode[None]) & mask[None]).sum(dim=(1, 2)) / mask.sum()
    )
    traj_acc = (
        ((pred == y_traj[None]) & mask[None]).sum(dim=(1, 2)) / mask.sum()
    )

    return base_acc.tolist(), mode_acc.tolist(), traj_acc.tolist()


def main():
    x_base, y_base = baseline_data()
    x_conf, y_conf, y_traj, meta = conflict_data()

    train_idx = [i for i, (_, start) in enumerate(meta) if start in (2, 3, 4)]
    hold_idx = [i for i, (_, start) in enumerate(meta) if start in (5, 6, 7)]

    x_target = torch.cat([x_base, x_conf[train_idx]], dim=0)
    y_target = torch.cat([y_base, y_conf[train_idx]], dim=0)

    # 120 extra copies: more sample count, zero new support.
    x_redundant = torch.cat([x_base] * 11, dim=0)
    y_redundant = torch.cat([y_base] * 11, dim=0)

    # 100 novel arbitrary histories. Exact canonical and exact-conflict sequences
    # are excluded; targets still obey the mode token.
    rng = np.random.default_rng(RANDOM_DATA_SEED)
    permutations = list(itertools.permutations(range(2, 8)))
    base_pairs = {(int(row[0]), tuple(row[1:].tolist())) for row in x_base}
    conflict_pairs = {(int(row[0]), tuple(row[1:].tolist())) for row in x_conf}
    candidates = [
        (mode, perm)
        for mode in (0, 1)
        for perm in permutations
        if (mode, perm) not in base_pairs and (mode, perm) not in conflict_pairs
    ]
    chosen = rng.choice(len(candidates), size=100, replace=False)
    rx, ry = [], []
    for idx in chosen:
        mode, perm = candidates[int(idx)]
        seq = [mode] + list(perm)
        rx.append(seq)
        ry.append(target_for_mode(seq, mode))

    x_random = torch.cat([x_base, torch.tensor(rx)], dim=0)
    y_random = torch.cat([y_base, torch.tensor(ry)], dim=0)

    conditions = {
        "baseline_12": (x_base, y_base),
        "redundant_plus_120": (x_redundant, y_redundant),
        "targeted_plus_6": (x_target, y_target),
        "random_novel_plus_100": (x_random, y_random),
    }

    result = {
        "settings": {
            "seeds": SEEDS,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "schedule_seed": SCHEDULE_SEED,
            "random_data_seed": RANDOM_DATA_SEED,
            "targeted_train_starts": [2, 3, 4],
            "heldout_conflict_starts": [5, 6, 7],
        },
        "conditions": {},
    }

    for name, (xd, yd) in conditions.items():
        model = train_condition(xd, yd)
        base_acc, mode_acc, traj_acc = evaluate(
            model,
            x_base,
            y_base,
            x_conf[hold_idx],
            y_conf[hold_idx],
            y_traj[hold_idx],
        )
        result["conditions"][name] = {
            "n_sequences": int(xd.shape[0]),
            "base_accuracy_by_seed": base_acc,
            "heldout_mode_accuracy_by_seed": mode_acc,
            "heldout_trajectory_accuracy_by_seed": traj_acc,
            "heldout_mode_mean": float(np.mean(mode_acc)),
            "heldout_trajectory_mean": float(np.mean(traj_acc)),
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
