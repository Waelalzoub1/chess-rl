"""Batched PUCT search: many independent game trees advance in lockstep so that
every simulation step is ONE network forward pass over all games (GPU friendly).

Conventions
- A node stores per-edge arrays: prior P, visit count N, total value W. W is
  kept from the perspective of the side to move AT THAT NODE, so selection is
  argmax(W/N + U) with no sign juggling, and backup flips the sign once per ply.
- Terminal values are from the perspective of the side to move at the leaf:
  -1 if it is checkmated, 0 for any draw.
- Repetitions: a leaf whose position already occurred earlier in the game or
  earlier on the current search path is scored as a draw. (Two-fold is treated
  as a draw inside the search, the usual engine convention; the game itself
  ends on the real threefold rule.)
"""

import math

import chess
import numpy as np
import torch

from .encode import encode_board, move_to_index


class Node:
    __slots__ = ("moves", "idxs", "P", "N", "W", "children")

    def __init__(self, moves, idxs, priors):
        self.moves = moves
        self.idxs = idxs
        self.P = priors
        self.N = np.zeros(len(moves), dtype=np.float32)
        self.W = np.zeros(len(moves), dtype=np.float32)
        self.children = [None] * len(moves)


def position_key(board):
    return board._transposition_key()


def legal_moves_folded(board):
    """Legal moves with underpromotions dropped (they share the queen-promotion index)."""
    return [m for m in board.legal_moves if m.promotion is None or m.promotion == chess.QUEEN]


def leaf_status(board, game_keys, path_keys):
    """Returns (terminal_value or None, legal_moves or None)."""
    key = position_key(board)
    if key in path_keys or key in game_keys:
        return 0.0, None
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0, None
    legal = legal_moves_folded(board)
    if not legal:
        return (-1.0 if board.is_check() else 0.0), None
    return None, legal


def make_node(board, legal, logits):
    flip = board.turn == chess.BLACK
    idxs = np.fromiter((move_to_index(m, flip) for m in legal), dtype=np.int64, count=len(legal))
    p = logits[idxs].astype(np.float32)
    p = np.exp(p - p.max())
    p /= p.sum()
    return Node(legal, idxs, p)


@torch.no_grad()
def evaluate(model, device, boards, amp=True):
    """One forward pass. Returns (logits [B,4096] float32 numpy, values [B] numpy)."""
    x = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(device, non_blocking=True).float()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        logits, value = model(x)
    return logits.float().cpu().numpy(), value.float().cpu().numpy()


def run_searches(model, device, boards, game_keys, n_sims, c_puct=1.5, add_noise=True,
                 dirichlet_alpha=0.3, noise_frac=0.25, rng=None):
    """Searches every board in `boards` with n_sims simulations each.

    boards:    list of chess.Board, none of them terminal
    game_keys: list of sets with the position keys already seen in each game (excluding the current position)
    Returns (roots, root_values): root Node per board and the search value for the side to move.
    """
    rng = rng or np.random.default_rng()
    logits, values = evaluate(model, device, boards)
    roots = []
    for i, b in enumerate(boards):
        root = make_node(b, legal_moves_folded(b), logits[i])
        if add_noise and len(root.moves) > 1:
            noise = rng.dirichlet([dirichlet_alpha] * len(root.moves)).astype(np.float32)
            root.P = (1 - noise_frac) * root.P + noise_frac * noise
        roots.append(root)
    scratch = [b.copy(stack=False) for b in boards]
    root_keys = [position_key(b) for b in boards]

    for _ in range(n_sims):
        pending = []                                    # (game, path, leaf parent, action, legal moves)
        for g, root in enumerate(roots):
            node, board, path = root, scratch[g], []
            path_keys = {root_keys[g]}
            while True:
                total = node.N.sum()
                u = c_puct * node.P * (math.sqrt(total + 1.0) / (1.0 + node.N))
                q = np.where(node.N > 0, node.W / np.maximum(node.N, 1.0), 0.0)
                a = int(np.argmax(q + u))
                path.append((node, a))
                board.push(node.moves[a])
                child = node.children[a]
                if child is None:
                    break
                path_keys.add(position_key(board))
                node = child
            value, legal = leaf_status(board, game_keys[g], path_keys)
            if value is None:
                pending.append((g, path, legal))
            else:
                _backup(path, value)
                for _ in path:
                    board.pop()
        if pending:
            leaf_logits, leaf_values = evaluate(model, device, [scratch[g] for g, _, _ in pending])
            for k, (g, path, legal) in enumerate(pending):
                parent, a = path[-1]
                parent.children[a] = make_node(scratch[g], legal, leaf_logits[k])
                _backup(path, float(leaf_values[k]))
                for _ in path:
                    scratch[g].pop()

    root_values = [float(r.W.sum() / max(r.N.sum(), 1.0)) for r in roots]
    return roots, root_values


def _backup(path, leaf_value):
    """leaf_value is from the perspective of the side to move at the leaf."""
    v = leaf_value
    for node, a in reversed(path):
        v = -v                                          # now from the perspective of the side that moved into it
        node.N[a] += 1.0
        node.W[a] += v


def choose_move(root, temperature, rng):
    """Samples from visit counts (temperature 0 = most visited). Returns (move, action index)."""
    if temperature < 1e-3:
        a = int(np.argmax(root.N))
    else:
        p = root.N.astype(np.float64) ** (1.0 / temperature)
        p /= p.sum()
        a = int(rng.choice(len(p), p=p))
    return root.moves[a], a


def visit_policy(root):
    """(policy indices, visit probabilities): the policy training target."""
    return root.idxs.astype(np.int16), (root.N / root.N.sum()).astype(np.float32)
