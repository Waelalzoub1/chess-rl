#!/usr/bin/env python3
"""Strength of a checkpoint against fixed baselines.

    python evaluate.py [--ckpt checkpoints/model.pt] [--games 100] [--sims 200] [--opponents random greedy]

The model plays with MCTS (no exploration noise, most-visited move), half the
games as white. `random` moves uniformly at random; `greedy` captures the most
valuable piece it can and otherwise moves at random. Prints W/D/L, the score
(win 1, draw 0.5) and a 95 % Wilson interval, and appends to --out if given.
"""
import argparse
import json
import time

import numpy as np
import torch

from chessrl.arena import play_match
from chessrl.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--opponents", nargs="+", default=["random", "greedy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    model, it = load_model(a.ckpt, device=device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    out = []
    for opp in a.opponents:
        t0 = time.time()
        r = play_match(model, device, opp, a.games, a.sims, rng)
        r.update({"ckpt_iter": it, "seconds": round(time.time() - t0)})
        out.append(r)
        print(f"iter {it} vs {opp:7s} games={r['games']} sims={r['sims']} W/D/L={r['wins']}/{r['draws']}/{r['losses']} "
              f"score={r['score']:.3f} wilson95=[{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}] ({r['seconds']}s)", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
