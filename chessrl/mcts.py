"""AlphaZero-style Monte Carlo Tree Search (PUCT).

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

import math

import numpy as np
import torch

from .encode import encode_board, legal_move_indices


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}  # policy index -> (Move, Node)

    def q(self):
        """Average value from the perspective of the side to move here."""
        return self.value_sum / self.visits if self.visits else 0.0

    def expanded(self):
        return bool(self.children)


def _evaluate(model, board):
    x = torch.from_numpy(encode_board(board)).float().unsqueeze(0)
    with torch.no_grad():
        logits, value = model(x)
    return logits[0].numpy(), float(value[0])


def _expand(node, board, logits):
    indices, idx_to_move = legal_move_indices(board)
    priors = logits[indices]
    priors = np.exp(priors - priors.max())
    priors /= priors.sum()
    for idx, p in zip(indices, priors):
        node.children[int(idx)] = (idx_to_move[int(idx)], Node(float(p)))


def _select_child(node, c_puct):
    sqrt_total = math.sqrt(node.visits)
    best, best_score = None, -1e9
    for idx, (move, child) in node.children.items():
        # child.q() is from the opponent's perspective, hence the negation
        u = c_puct * child.prior * sqrt_total / (1 + child.visits)
        score = -child.q() + u
        if score > best_score:
            best_score, best = score, (move, child)
    return best


def _terminal_value(board):
    """Value from the mover's perspective if the position is terminal."""
    if board.is_checkmate():
        return -1.0  # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.halfmove_clock >= 100:
        return 0.0
    return None


def search(model, board, n_sims, c_puct=1.5, dirichlet_alpha=0.3,
           noise_frac=0.25, add_noise=True, rng=None):
    """Runs n_sims simulations. Returns (visits, moves, root_q):
    visits/moves map policy index -> visit count / Move, root_q is the
    search value estimate from the mover's perspective."""
    rng = rng or np.random
    root = Node(0.0)
    logits, root_value = _evaluate(model, board)
    _expand(root, board, logits)
    if add_noise and len(root.children) > 1:
        noise = rng.dirichlet([dirichlet_alpha] * len(root.children))
        for (_, (_, child)), n in zip(root.children.items(), noise):
            child.prior = (1 - noise_frac) * child.prior + noise_frac * float(n)
    root.visits, root.value_sum = 1, root_value

    for _ in range(n_sims):
        node, path = root, [root]
        scratch = board.copy(stack=False)
        while node.expanded():
            move, node = _select_child(node, c_puct)
            scratch.push(move)
            path.append(node)
        value = _terminal_value(scratch)
        if value is None:
            leaf_logits, value = _evaluate(model, scratch)
            _expand(node, scratch, leaf_logits)
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value
            value = -value

    visits = {idx: child.visits for idx, (_, child) in root.children.items()}
    moves = {idx: move for idx, (move, _) in root.children.items()}
    return visits, moves, root.q()


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
    """Normalized visit distribution — the policy training target.
    Returns (indices, probs) as a sparse pair."""
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.array([visits[int(i)] for i in idxs], dtype=np.float32)
    return idxs, counts / counts.sum()
