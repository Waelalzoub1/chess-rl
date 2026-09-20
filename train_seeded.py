"""Parallel self-play RL training.

Each iteration: N worker processes play MCTS self-play games against the
current checkpoint in parallel, then the main process trains the network to
match the search's visit distributions (policy) and the game outcomes
(value) — the AlphaZero recipe — and saves a new checkpoint. watch.py always
picks up the latest checkpoint, so you can watch strength evolve mid-run.

Usage: python train.py [--iters 200] [--workers 7] [--games-per-worker 8]
"""

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

from chessrl.encode import POLICY_SIZE
from chessrl.model import PolicyValueNet, load_model
from chessrl.selfplay import run_selfplay_batch

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model.pt")
STATS = os.path.join(os.path.dirname(CKPT), "stats.json")



def save_ckpt(model, iteration):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + ".tmp"
    torch.save({"model": model.state_dict(), "iter": iteration}, tmp)
    os.replace(tmp, CKPT)


def append_stats(entry):
    history = []
    if os.path.exists(STATS):
        with open(STATS) as f:
            history = json.load(f)
    history.append(entry)
    with open(STATS, "w") as f:
        json.dump(history, f)


def train_on(model, opt, states, pi_sparse, rewards, epochs, batch_size):
    """AlphaZero-style update: policy learns the MCTS visit distribution,
    value learns the game outcome."""
    model.train()
    n = len(states)
    p_losses, v_losses = [], []
    for _ in range(epochs):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            x = torch.from_numpy(states[idx]).float()
            z = torch.from_numpy(rewards[idx])
            pi = torch.zeros(len(idx), POLICY_SIZE)
            for row, j in enumerate(idx):
                cols, probs = pi_sparse[j]
                pi[row, cols] = torch.from_numpy(probs)
            logits, value = model(x)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z)
            loss = policy_loss + 0.5 * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
    model.eval()
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--games-per-worker", type=int, default=4)
    ap.add_argument("--sims", type=int, default=48,
                    help="MCTS simulations per move during self-play")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=None, help="fixed seed for numpy/torch and the worker seeds")
    ap.add_argument("--out-dir", default=None, help="checkpoint directory (default: checkpoints/)")
    ap.add_argument("--no-shaping", action="store_true", help="ablation: truncated games get reward 0 instead of the material-based shaped reward")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint in out-dir")
    args = ap.parse_args()
    global CKPT, STATS
    if args.out_dir:
        CKPT = os.path.join(args.out_dir, "model.pt"); STATS = os.path.join(args.out_dir, "stats.json")
    if args.seed is not None:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.no_shaping:
        import chessrl.selfplay as sp
        sp.SHAPING = 0.0
    if args.fresh and os.path.exists(CKPT):
        os.remove(CKPT)
        if os.path.exists(STATS): os.remove(STATS)

    torch.set_num_threads(max(1, os.cpu_count() - args.workers))
    model, start_iter = load_model(CKPT)
    if start_iter:
        print(f"resuming from iteration {start_iter}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_ckpt(model, start_iter)

    ctx = get_context("spawn")
    pool = ctx.Pool(args.workers)
    for it in range(start_iter + 1, start_iter + 1 + args.iters):
        t0 = time.time()
        seeds = np.random.randint(0, 2**31 - 1, size=args.workers)
        jobs = [(CKPT, args.games_per_worker, int(s), args.max_plies, args.sims, 0.0 if args.no_shaping else 0.05)
                for s in seeds]
        results = pool.map(run_selfplay_batch, jobs)
        t_play = time.time() - t0

        states = np.concatenate([r[0] for r in results])
        pi_sparse = [p for r in results for p in r[1]]
        rewards = np.concatenate([r[2] for r in results])
        agg = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "trunc": 0, "plies": 0}
        for r in results:
            for k in agg:
                agg[k] += r[3][k]
        n_games = args.workers * args.games_per_worker

        p_loss, v_loss = train_on(model, opt, states, pi_sparse, rewards,
                                  args.epochs, args.batch_size)
        save_ckpt(model, it)

        entry = {
            "iter": it,
            "games": n_games,
            "white_wins": agg["1-0"],
            "black_wins": agg["0-1"],
            "draws": agg["1/2-1/2"],
            "truncated": agg["trunc"],
            "avg_plies": round(agg["plies"] / n_games, 1),
            "positions": int(len(states)),
            "policy_loss": round(p_loss, 4),
            "value_loss": round(v_loss, 4),
            "seconds": round(time.time() - t0, 1),
        }
        append_stats(entry)
        print(f"[iter {it}] games={n_games} "
              f"W/B/D/T={agg['1-0']}/{agg['0-1']}/{agg['1/2-1/2']}/{agg['trunc']} "
              f"avg_plies={entry['avg_plies']} pos={entry['positions']} "
              f"p_loss={p_loss:.4f} v_loss={v_loss:.4f} "
              f"({t_play:.1f}s play, {entry['seconds']}s total)", flush=True)


if __name__ == "__main__":
    main()
