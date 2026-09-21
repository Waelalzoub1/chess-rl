"""Matches against fixed baselines, played concurrently with batched search."""

import math

import chess
import numpy as np

from .batched import choose_move, position_key, run_searches

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def random_mover(board, rng):
    moves = list(board.legal_moves)
    return moves[int(rng.integers(len(moves)))]


def greedy_capture_mover(board, rng):
    """Takes the most valuable piece it can capture (ties at random); otherwise moves at random."""
    best, best_value = [], 0
    for m in board.legal_moves:
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or chess.PAWN      # en passant
            v = VALUE[victim]
            if v > best_value:
                best, best_value = [m], v
            elif v == best_value:
                best.append(m)
    if best:
        return best[int(rng.integers(len(best)))]
    return random_mover(board, rng)


OPPONENTS = {"random": random_mover, "greedy": greedy_capture_mover}


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def play_match(model, device, opponent, n_games, n_sims, rng, max_plies=300):
    """The model (MCTS, n_sims per move, no noise, most-visited move) against `opponent`.
    Half the games as white. A game is a draw at threefold repetition, the 50-move rule,
    stalemate, insufficient material, or after max_plies. Returns dict with W/D/L and score."""
    mover = OPPONENTS[opponent]
    games = [{"board": chess.Board(), "keys": set(), "white": i % 2 == 0} for i in range(n_games)]
    results = []
    active = games
    while active:
        to_search = [g for g in active if g["board"].turn == (chess.WHITE if g["white"] else chess.BLACK)]
        if to_search:
            roots, _ = run_searches(model, device, [g["board"] for g in to_search], [g["keys"] for g in to_search],
                                    n_sims, add_noise=False, rng=rng)
            chosen = {id(g): choose_move(r, 0.0, rng)[0] for g, r in zip(to_search, roots)}
        else:
            chosen = {}
        still = []
        for g in active:
            b = g["board"]
            g["keys"].add(position_key(b))
            b.push(chosen[id(g)] if id(g) in chosen else mover(b, rng))
            if b.is_checkmate():
                model_won = (b.turn == chess.BLACK) == g["white"]
                results.append(1.0 if model_won else 0.0)
            elif (b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100
                  or b.is_repetition(3) or b.ply() >= max_plies):
                results.append(0.5)
            else:
                still.append(g)
        active = still
    w, d, l = results.count(1.0), results.count(0.5), results.count(0.0)
    score = (w + 0.5 * d) / n_games
    lo, hi = wilson(score, n_games)
    return {"opponent": opponent, "games": n_games, "sims": n_sims, "wins": w, "draws": d, "losses": l,
            "score": score, "wilson95": [lo, hi]}
