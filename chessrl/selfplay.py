"""Self-play game generation via MCTS. `run_selfplay_batch` is the entry
point used by the parallel worker processes spawned from train.py."""

import chess
import numpy as np
import torch

from .encode import encode_board, legal_move_indices, material_balance
from .mcts import search, select_move, visit_policy
from .model import load_model


def pick_move(model, board, temperature, rng=None):
    """Samples a move from the raw (searchless) masked policy.
    Returns (move, index, value)."""
    x = torch.from_numpy(encode_board(board)).float().unsqueeze(0)
    with torch.no_grad():
        logits, value = model(x)
    logits = logits[0].numpy()
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


def _game_over(board, max_plies):
    return (board.is_checkmate() or board.is_stalemate() or
            board.is_insufficient_material() or board.halfmove_clock >= 100 or
            board.ply() >= max_plies)


def play_game(model, n_sims=48, max_plies=160, opening_temp=1.0,
              temp_after=30, late_temp=0.25, rng=None, shaping=0.05):
    """Plays one MCTS self-play game.

    Returns (states, pi_sparse, rewards, info): pi_sparse is a list of
    (indices, probs) visit-count policy targets, rewards are the final
    outcome from the mover's perspective at each position.
    """
    board = chess.Board()
    states, pi_sparse, movers = [], [], []
    while not _game_over(board, max_plies):
        visits, moves, _ = search(model, board, n_sims, add_noise=True, rng=rng)
        temp = opening_temp if board.ply() < temp_after else late_temp
        move, _ = select_move(visits, moves, temp, rng)
        states.append(encode_board(board))
        pi_sparse.append(visit_policy(visits))
        movers.append(board.turn)
        board.push(move)

    truncated = False
    if board.is_checkmate():
        z_white = 1.0 if board.turn == chess.BLACK else -1.0
        outcome = "1-0" if z_white > 0 else "0-1"
    elif board.ply() >= max_plies:
        # Truncated: shaped reward from material so early training still gets
        # a gradient out of the many long aimless games.
        truncated = True
        z_white = float(np.clip(material_balance(board) * shaping, -0.5, 0.5))
        outcome = "trunc"
    else:
        z_white = 0.0
        outcome = "1/2-1/2"

    rewards = np.array(
        [z_white if mover == chess.WHITE else -z_white for mover in movers],
        dtype=np.float32,
    )
    info = {"outcome": outcome, "plies": board.ply(), "truncated": truncated}
    return np.stack(states), pi_sparse, rewards, info


def run_selfplay_batch(args):
    """Worker entry point:
    (ckpt_path, n_games, seed, max_plies, n_sims) -> samples."""
    ckpt_path, n_games, seed, max_plies, n_sims = args[:5]
    shaping = args[5] if len(args) > 5 else 0.05
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model, _ = load_model(ckpt_path)

    all_states, all_pi, all_rewards = [], [], []
    stats = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "trunc": 0, "plies": 0}
    for _ in range(n_games):
        states, pi_sparse, rewards, info = play_game(
            model, n_sims=n_sims, max_plies=max_plies, rng=rng, shaping=shaping)
        all_states.append(states)
        all_pi.extend(pi_sparse)
        all_rewards.append(rewards)
        stats[info["outcome"]] += 1
        stats["plies"] += info["plies"]
    return (
        np.concatenate(all_states),
        all_pi,
        np.concatenate(all_rewards),
        stats,
    )
