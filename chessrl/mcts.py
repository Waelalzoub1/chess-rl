"""AlphaZero-style Monte Carlo Tree Search (PUCT): single-position API.

Each search builds a tree rooted at the current position. A simulation walks
down the tree picking the child with the best score = Q + U, where Q is the
average outcome seen through that child so far and U is an exploration bonus
proportional to the network's prior for the move and shrinking with visits.
At the leaf, the network evaluates the position (or the rules do, if it's
terminal) and that value is backed up the path, flipping sign at every ply
since the players alternate. After N simulations the root's visit counts are
a better policy than the raw network output — that's what gets played and
what the network is trained to imitate.
"""

import numpy as np
import torch

from . import batched


def _history_keys(board):
    """Position keys of every earlier position of this game (for repetition detection)."""
    keys, b = set(), board.copy()
    while b.move_stack:
        b.pop()
        keys.add(batched.position_key(b))
    return keys


def search(model, board, n_sims, c_puct=1.5, dirichlet_alpha=0.3,
           noise_frac=0.25, add_noise=True, rng=None):
    """Runs n_sims simulations from one position. Returns (visits, moves, root_q):
    visits/moves map policy index -> visit count / Move, root_q is the search
    value estimate from the mover's perspective. Thin wrapper over the batched
    search in batched.py (a batch of one)."""
    if rng is None or not hasattr(rng, "dirichlet") or not hasattr(rng, "choice"):
        rng = np.random.default_rng()
    device = next(model.parameters()).device
    roots, values = batched.run_searches(model, device, [board], [_history_keys(board)], n_sims, c_puct=c_puct,
                                         add_noise=add_noise, dirichlet_alpha=dirichlet_alpha,
                                         noise_frac=noise_frac, rng=rng)
    root = roots[0]
    visits = {int(i): int(n) for i, n in zip(root.idxs, root.N)}
    moves = {int(i): m for i, m in zip(root.idxs, root.moves)}
    return visits, moves, values[0]


def select_move(visits, moves, temperature, rng=None):
    """Picks a move from root visit counts. Returns (move, index)."""
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.array([visits[int(i)] for i in idxs], dtype=np.float64)
    if temperature < 1e-3:
        choice = idxs[int(np.argmax(counts))]
    else:
        probs = counts ** (1.0 / temperature)
        probs /= probs.sum()
        rng = rng or np.random
        choice = rng.choice(idxs, p=probs)
    return moves[int(choice)], int(choice)


def visit_policy(visits):
    """Normalized visit distribution. Returns (indices, probs) as a sparse pair."""
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.array([visits[int(i)] for i in idxs], dtype=np.float32)
    return idxs, counts / counts.sum()
