#!/usr/bin/env python3
"""Measure a checkpoint's strength against a uniformly random mover.

    python eval_vs_random.py [--ckpt checkpoints/model.pt] [--games 100] [--sims 0] [--seed 0]

--sims 0 plays the raw policy head (argmax over legal moves); --sims N uses
MCTS with N simulations per move. Half the games are played as white, half as
black. Prints wins/draws/losses, the score (win=1, draw=0.5), and a Wilson
95% interval. Random-vs-random is also reported as the baseline.
"""
import argparse, math, sys, os, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chessrl.model import load_model
from chessrl.selfplay import pick_move
from chessrl.mcts import search, select_move

def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def model_move(model, board, sims, rng):
    if sims <= 0:
        move, _, _ = pick_move(model, board, temperature=0.0, rng=rng)
        return move
    visits, moves, _ = search(model, board, sims, add_noise=False, rng=rng)
    move, _ = select_move(visits, moves, 0.0, rng)
    return move

def play(model, model_is_white, sims, rng, max_plies=200):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == chess.WHITE and model_is_white or board.turn == chess.BLACK and not model_is_white:
            if model is None: move = rng.choice(list(board.legal_moves))
            else: move = model_move(model, board, sims, rng)
        else:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
    r = board.result(claim_draw=True)
    if r == '*': return 0.5
    if r == '1/2-1/2': return 0.5
    return 1.0 if (r == '1-0') == model_is_white else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/model.pt'); ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--sims', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--baseline', action='store_true', help='also play random vs random')
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model, it = load_model(a.ckpt) if a.ckpt != 'none' else (None, 0)
    t0 = time.time(); scores = [play(model, g % 2 == 0, a.sims, rng) for g in range(a.games)]
    w, d, l = scores.count(1.0), scores.count(0.5), scores.count(0.0)
    p = sum(scores) / len(scores); lo, hi = wilson(p, len(scores))
    print(f'ckpt={a.ckpt} iter={it} sims={a.sims} games={a.games} W/D/L={w}/{d}/{l} score={p:.3f} wilson95=[{lo:.3f},{hi:.3f}] wins={w/len(scores):.3f} ({time.time()-t0:.0f}s)')
    if a.baseline:
        scores = [play(None, g % 2 == 0, 0, rng) for g in range(a.games)]
        w, d, l = scores.count(1.0), scores.count(0.5), scores.count(0.0)
        print(f'random-vs-random games={a.games} W/D/L={w}/{d}/{l} score={sum(scores)/len(scores):.3f}')

if __name__ == '__main__':
    main()
