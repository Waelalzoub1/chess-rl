"""AlphaZero-style training.

Each iteration: actor processes play batched MCTS self-play on the GPU against
the latest checkpoint; their positions go into a replay buffer; the learner
takes gradient steps on minibatches sampled from the buffer (policy learns the
search's visit distribution, value learns the game outcome) and writes a new
checkpoint. watch.py always picks up the latest checkpoint.

Usage: python train.py [--iters 40] [--actors 7] [--games-per-actor 32] [--sims 64]
"""

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

from chessrl.arena import play_match
from chessrl.encode import POLICY_SIZE
from chessrl.model import PolicyValueNet, load_model
from chessrl.selfplay import actor_job

HERE = os.path.dirname(os.path.abspath(__file__))


class ReplayBuffer:
    """Ring buffer of (state, sparse policy target, outcome) over the most recent positions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.zeros((capacity, 17, 8, 8), dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.pis = [None] * capacity
        self.size = 0
        self.pos = 0

    def add(self, states, pis, rewards):
        for s, p, z in zip(states, pis, rewards):
            self.states[self.pos], self.pis[self.pos], self.rewards[self.pos] = s, p, z
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device):
        idx = rng.integers(0, self.size, size=batch_size)
        x = torch.from_numpy(self.states[idx]).to(device).float()
        z = torch.from_numpy(self.rewards[idx]).to(device)
        pi = torch.zeros(batch_size, POLICY_SIZE)
        for row, j in enumerate(idx):
            cols, probs = self.pis[j]
            pi[row, cols.astype(np.int64)] = torch.from_numpy(probs)
        return x, pi.to(device), z


def save_ckpt(model, iteration, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "iter": iteration, "channels": model.channels, "blocks": model.n_blocks}, tmp)
    os.replace(tmp, path)


def train_steps(model, opt, buffer, steps, batch_size, rng, device):
    model.train()
    p_losses, v_losses = [], []
    for _ in range(steps):
        x, pi, z = buffer.sample(batch_size, rng, device)
        logits, value = model(x)
        policy_loss = -(pi * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, z)
        loss = policy_loss + value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        p_losses.append(policy_loss.item())
        v_losses.append(value_loss.item())
    model.eval()
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--actors", type=int, default=7, help="self-play processes (each shares the GPU)")
    ap.add_argument("--games-per-actor", type=int, default=32)
    ap.add_argument("--concurrent", type=int, default=32, help="games searched in one batch per actor")
    ap.add_argument("--sims", type=int, default=64, help="MCTS simulations per move during self-play")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--buffer", type=int, default=400000, help="replay buffer capacity in positions")
    ap.add_argument("--train-steps", type=int, default=300, help="gradient steps per iteration")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--no-shaping", action="store_true",
                    help="ablation: truncated games get reward 0 instead of the material-shaped reward")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint in out-dir")
    ap.add_argument("--eval-every", type=int, default=5, help="play a quick match vs random every N iterations (0 = off)")
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-sims", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = os.path.join(args.out_dir, "model.pt")
    stats_path = os.path.join(args.out_dir, "stats.json")
    if args.fresh:
        for p in (ckpt, stats_path):
            if os.path.exists(p):
                os.remove(p)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    shaping = 0.0 if args.no_shaping else 0.05

    model, start_iter = load_model(ckpt, device=device, channels=args.channels, blocks=args.blocks)
    print(f"{'resuming from iteration ' + str(start_iter) if start_iter else 'fresh network'}: "
          f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    save_ckpt(model, start_iter, ckpt)
    buffer = ReplayBuffer(args.buffer)
    history = json.load(open(stats_path)) if os.path.exists(stats_path) else []

    pool = get_context("spawn").Pool(args.actors)
    for it in range(start_iter + 1, start_iter + 1 + args.iters):
        t0 = time.time()
        seeds = rng.integers(0, 2**31 - 1, size=args.actors)
        jobs = [(ckpt, args.games_per_actor, args.concurrent, int(s), args.max_plies, args.sims, shaping, args.device)
                for s in seeds]
        results = pool.map(actor_job, jobs)
        t_play = time.time() - t0
        agg = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "trunc": 0, "plies": 0}
        n_pos = 0
        for states, pis, rewards, stats in results:
            buffer.add(states, pis, rewards)
            n_pos += len(states)
            for k in agg:
                agg[k] += stats[k]
        n_games = args.actors * args.games_per_actor
        p_loss, v_loss = train_steps(model, opt, buffer, args.train_steps, args.batch_size, rng, device)
        save_ckpt(model, it, ckpt)

        entry = {"iter": it, "games": n_games, "white_wins": agg["1-0"], "black_wins": agg["0-1"],
                 "draws": agg["1/2-1/2"], "truncated": agg["trunc"], "avg_plies": round(agg["plies"] / n_games, 1),
                 "positions": n_pos, "buffer": buffer.size, "policy_loss": round(p_loss, 4),
                 "value_loss": round(v_loss, 4), "play_seconds": round(t_play, 1)}
        if args.eval_every and it % args.eval_every == 0:
            match = play_match(model, device, "random", args.eval_games, args.eval_sims, rng)
            entry["vs_random"] = {k: match[k] for k in ("wins", "draws", "losses", "score")}
        entry["seconds"] = round(time.time() - t0, 1)
        history.append(entry)
        with open(stats_path, "w") as f:
            json.dump(history, f)
        print(f"[iter {it}] games={n_games} W/B/D/T={agg['1-0']}/{agg['0-1']}/{agg['1/2-1/2']}/{agg['trunc']} "
              f"plies={entry['avg_plies']} pos={n_pos} buf={buffer.size} p_loss={p_loss:.4f} v_loss={v_loss:.4f} "
              f"({t_play:.0f}s play, {entry['seconds']}s total)"
              + (f" vs_random={entry['vs_random']}" if "vs_random" in entry else ""), flush=True)


if __name__ == "__main__":
    main()
