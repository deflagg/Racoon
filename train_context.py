from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F

from raccoon import Raccoon, RaccoonConfig, count_parameters


FORWARD = [2, 3, 4, 5, 6, 7]
BACKWARD = [2, 7, 6, 5, 4, 3]


def cycle_sequence(mode: int, start: int) -> list[int]:
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    if start not in range(2, 8):
        raise ValueError("start must be one of 2..7")

    cycle = FORWARD if mode == 0 else BACKWARD
    i = cycle.index(start)
    ordered = cycle[i:] + cycle[:i]
    return [mode] + ordered


def make_dataset(device: torch.device):
    """All 12 mode/start combinations.

    Input has length 7: [mode, six cycle tokens].

    We ignore the loss at position 0 because the first cycle token is chosen
    independently of the mode. From position 1 onward, the correct next token
    depends on both the current token and the earlier mode token.
    """
    xs = []
    ys = []

    for mode in (0, 1):
        cycle = FORWARD if mode == 0 else BACKWARD
        for start in range(2, 8):
            seq = cycle_sequence(mode, start)

            # Predict the successor of each cycle token.
            target = [-100]
            for token in seq[1:]:
                j = cycle.index(token)
                target.append(cycle[(j + 1) % len(cycle)])

            xs.append(seq)
            ys.append(target)

    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(model: Raccoon, x: torch.Tensor, y: torch.Tensor, *, ablate_attention=False):
    model.eval()
    logits = model(x, ablate_attention=ablate_attention)
    pred = logits.argmax(dim=-1)

    mask = y != -100
    correct = (pred[mask] == y[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
    ).item()

    return {"loss": loss, "accuracy": accuracy, "correct": correct, "total": total}


def grad_norm(model: Raccoon) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = RaccoonConfig()
    model = Raccoon(cfg).to(device)
    assert count_parameters(model) == 244

    x, y = make_dataset(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []

    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(model, x, y)
            record = {"step": step, **metrics}
            history.append(record)
            print(json.dumps(record))

        if step == args.steps:
            break

        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model)
        optimizer.step()

        if step % args.log_every == 0:
            print(json.dumps({"step": step, "train_grad_norm": gnorm}))

    final = evaluate(model, x, y)
    ablated = evaluate(model, x, y, ablate_attention=True)

    summary = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "config": asdict(cfg),
        "parameters": count_parameters(model),
        "final": final,
        "attention_ablated": ablated,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
