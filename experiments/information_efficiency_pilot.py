from __future__ import annotations

import itertools
import json

import numpy as np
import torch

import experiments.data_curation_pilot as pilot

# Compute-matched pilot settings.
pilot.SEEDS = [0, 1, 2]
pilot.STEPS = 1500
pilot.BATCH_SIZE = 24
pilot.LR = 3e-3
pilot.SCHEDULE_SEED = 999
RANDOM_DATA_SEED = 20261013

TARGET_ORDER = [0, 6, 1, 7, 2, 8]  # interleave mode 0/1 at starts 2,3,4
HOLD_IDX = [3, 4, 5, 9, 10, 11]    # held-out conflict starts 5,6,7


@torch.no_grad()
def evaluate_rule(model, x_base, y_base, x_hold, y_mode, y_traj):
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
    return base_acc, mode_acc, traj_acc


def main():
    x_base, y_base = pilot.baseline_data()
    x_conf, y_conf, y_traj, meta = pilot.conflict_data()

    permutations = list(itertools.permutations(range(2, 8)))
    base_pairs = {(int(r[0]), tuple(r[1:].tolist())) for r in x_base}
    conflict_pairs = {(int(r[0]), tuple(r[1:].tolist())) for r in x_conf}

    candidates = {0: [], 1: []}
    for mode in (0, 1):
        for perm in permutations:
            if (mode, perm) not in base_pairs and (mode, perm) not in conflict_pairs:
                candidates[mode].append(perm)

    rng = np.random.default_rng(RANDOM_DATA_SEED)
    for mode in (0, 1):
        rng.shuffle(candidates[mode])

    random_examples = []
    for j in range(3):
        for mode in (0, 1):
            seq = [mode] + list(candidates[mode][j])
            random_examples.append((seq, pilot.target_for_mode(seq, mode)))

    conditions = [("baseline", 0, x_base, y_base)]
    for k in range(1, 7):
        idx = TARGET_ORDER[:k]
        conditions.append((
            "targeted", k,
            torch.cat([x_base, x_conf[idx]], dim=0),
            torch.cat([y_base, y_conf[idx]], dim=0),
        ))
    for k in range(1, 7):
        rx = torch.tensor([x for x, _ in random_examples[:k]])
        ry = torch.tensor([y for _, y in random_examples[:k]])
        conditions.append((
            "random", k,
            torch.cat([x_base, rx], dim=0),
            torch.cat([y_base, ry], dim=0),
        ))

    rows = []
    for family, k, x, y in conditions:
        model = pilot.train_condition(x, y)
        base_acc, mode_acc, traj_acc = evaluate_rule(
            model,
            x_base,
            y_base,
            x_conf[HOLD_IDX],
            y_conf[HOLD_IDX],
            y_traj[HOLD_IDX],
        )
        rows.append({
            "family": family,
            "k": k,
            "n_sequences": int(x.shape[0]),
            "base_accuracy_by_seed": base_acc.tolist(),
            "heldout_mode_accuracy_by_seed": mode_acc.tolist(),
            "heldout_trajectory_accuracy_by_seed": traj_acc.tolist(),
            "base_mean": float(base_acc.mean()),
            "heldout_mode_mean": float(mode_acc.mean()),
            "heldout_trajectory_mean": float(traj_acc.mean()),
        })

    print(json.dumps({
        "settings": {
            "seeds": pilot.SEEDS,
            "steps": pilot.STEPS,
            "batch_size": pilot.BATCH_SIZE,
            "lr": pilot.LR,
            "schedule_seed": pilot.SCHEDULE_SEED,
            "random_data_seed": RANDOM_DATA_SEED,
            "target_order": [meta[i] for i in TARGET_ORDER],
            "heldout_conflicts": [meta[i] for i in HOLD_IDX],
        },
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
