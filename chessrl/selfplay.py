"""Self-play game generation. Many games are played concurrently so every
search step is one batched network call (see batched.py). `actor_job` is the
entry point for the actor processes started by train.py."""

import chess
import numpy as np
import torch

from .batched import choose_move, position_key, run_searches, visit_policy
from .encode import encode_board, legal_move_indices, material_balance
from .model import load_model


def pick_move(model, board, temperature, rng=None):
    """Samples a move from the raw (searchless) masked policy.
    Returns (move, index, value)."""
    device = next(model.parameters()).device
    x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(x)
    logits = logits[0].float().cpu().numpy()
    indices, idx_to_move = legal_move_indices(board)
    legal_logits = logits[indices]
    if temperature < 1e-3:
        choice = indices[int(np.argmax(legal_logits))]
    else:
        z = (legal_logits - legal_logits.max()) / temperature
        probs = np.exp(z)
        probs /= probs.sum()
        rng = rng or np.random
        choice = rng.choice(indices, p=probs)
    return idx_to_move[int(choice)], int(choice), float(value[0])


def game_result(board, max_plies, shaping):
    """None while the game goes on, else (z_white, label)."""
    if board.is_checkmate():
        return (1.0, "1-0") if board.turn == chess.BLACK else (-1.0, "0-1")
    if (board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(3)):
        return 0.0, "1/2-1/2"
    if board.ply() >= max_plies:
        # Truncated: shaped reward from material so long aimless games still carry a signal.
        return float(np.clip(material_balance(board) * shaping, -0.5, 0.5)), "trunc"
    return None


class _Game:
    __slots__ = ("board", "keys", "states", "pis", "movers")

    def __init__(self):
        self.board = chess.Board()
        self.keys = set()
        self.states, self.pis, self.movers = [], [], []


def play_games(model, device, n_games, concurrent, n_sims, max_plies=160, shaping=0.05, rng=None,
               opening_temp=1.0, temp_after=30, late_temp=0.25):
    """Plays n_games of self-play, `concurrent` at a time.
    Returns (states uint8 [N,17,8,8], list of (policy idxs, probs), rewards float32 [N], stats)."""
    rng = rng or np.random.default_rng()
    stats = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "trunc": 0, "plies": 0}
    states, pis, rewards = [], [], []
    active, started = [], 0
    while active or started < n_games:
        while len(active) < concurrent and started < n_games:
            active.append(_Game())
            started += 1
        roots, _ = run_searches(model, device, [g.board for g in active], [g.keys for g in active],
                                n_sims, add_noise=True, rng=rng)
        still = []
        for g, root in zip(active, roots):
            temp = opening_temp if g.board.ply() < temp_after else late_temp
            move, _ = choose_move(root, temp, rng)
            g.states.append(encode_board(g.board))
            g.pis.append(visit_policy(root))
            g.movers.append(g.board.turn)
            g.keys.add(position_key(g.board))
            g.board.push(move)
            result = game_result(g.board, max_plies, shaping)
            if result is None:
                still.append(g)
                continue
            z_white, label = result
            stats[label] += 1
            stats["plies"] += g.board.ply()
            states.extend(g.states)
            pis.extend(g.pis)
            rewards.extend(z_white if m == chess.WHITE else -z_white for m in g.movers)
        active = still
    return np.stack(states), pis, np.asarray(rewards, dtype=np.float32), stats


def actor_job(args):
    """(ckpt_path, n_games, concurrent, seed, max_plies, n_sims, shaping, device) -> samples."""
    ckpt_path, n_games, concurrent, seed, max_plies, n_sims, shaping, device = args
    torch.set_num_threads(1)
    device = torch.device(device)
    model, _ = load_model(ckpt_path, device=device)
    return play_games(model, device, n_games, concurrent, n_sims, max_plies=max_plies, shaping=shaping,
                      rng=np.random.default_rng(seed))
