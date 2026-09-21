"""Unit tests for the encoding, the value sign convention, and the search.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chessrl.encode import N_PLANES, POLICY_SIZE, encode_board, legal_move_indices, move_to_index
from chessrl.mcts import search, select_move
from chessrl.model import PolicyValueNet

MATE_IN_1 = [
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),          # back-rank mate, white to move
    ("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),          # the same idea with black to move (mirrored encoding)
    ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7"),             # queen mate supported by the king (one of several)
]


def random_positions(n, seed=0, max_plies=60):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(int(rng.integers(0, max_plies))):
            if b.is_game_over():
                break
            moves = list(b.legal_moves)
            b.push(moves[int(rng.integers(len(moves)))])
        if not b.is_game_over():
            out.append(b)
    return out


class EncodingTest(unittest.TestCase):
    def test_move_index_round_trip(self):
        for b in random_positions(200):
            indices, idx_to_move = legal_move_indices(b)
            self.assertEqual(len(set(indices.tolist())), len(indices))
            flip = b.turn == chess.BLACK
            for move in b.legal_moves:
                if move.promotion not in (None, chess.QUEEN):
                    continue                                   # underpromotions are folded into the queen index
                idx = move_to_index(move, flip)
                self.assertTrue(0 <= idx < POLICY_SIZE)
                self.assertEqual(idx_to_move[idx], move)

    def test_planes_are_from_the_movers_perspective(self):
        for b in random_positions(100, seed=1):
            x = encode_board(b)
            self.assertEqual(x.shape, (N_PLANES, 8, 8))
            m = b.mirror()                                     # swap colours and flip the board: same position for the other side
            self.assertTrue(np.array_equal(x, encode_board(m)))
            own_king = x[5]
            self.assertEqual(int(own_king.sum()), 1)

    def test_mirrored_position_gives_mirrored_move_index(self):
        for b in random_positions(50, seed=2):
            m = b.mirror()
            for move in b.legal_moves:
                mm = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
                self.assertEqual(move_to_index(move, b.turn == chess.BLACK), move_to_index(mm, m.turn == chess.BLACK))


class SearchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = PolicyValueNet().eval()                   # untrained: search must find mates from the rules alone

    def test_finds_mate_in_one(self):
        for fen, _ in MATE_IN_1:
            board = chess.Board(fen)
            visits, moves, root_q = search(self.model, board, 300, add_noise=False, rng=np.random.default_rng(0))
            move, _ = select_move(visits, moves, 0.0)
            board.push(move)
            self.assertTrue(board.is_checkmate(), f"{fen}: played {move}, not mate")

    def test_value_sign(self):
        # The side to move can mate: the search value for the mover must be clearly positive ...
        fen, _ = MATE_IN_1[0]
        _, _, q_winning = search(self.model, chess.Board(fen), 300, add_noise=False, rng=np.random.default_rng(0))
        self.assertGreater(q_winning, 0.2)
        # ... and one ply earlier, for the side that cannot avoid being mated, clearly negative.
        forced = chess.Board("7k/R7/6K1/8/8/8/8/8 b - - 0 1")      # black: only Kg8, then Ra8# is mate
        _, _, q_losing = search(self.model, forced, 400, add_noise=False, rng=np.random.default_rng(0))
        self.assertLess(q_losing, -0.2)


class RepetitionTest(unittest.TestCase):
    def test_search_scores_a_repeated_position_as_a_draw(self):
        from chessrl import batched
        # K+R vs K, white to move and winning. Shuffle Ra1-a2-a1 and the black king so that the
        # position after Ra2 has already occurred in the game.
        b = chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 0 1")
        for uci in ["a1a2", "h8g8", "a2a1", "g8h8"]:
            b.push_uci(uci)
        keys = set()
        h = b.copy()
        while h.move_stack:
            h.pop()
            keys.add(batched.position_key(h))
        c = b.copy(stack=False)
        c.push_uci("a1a2")                                   # recreates an earlier position
        value, legal = batched.leaf_status(c, keys, {batched.position_key(b)})
        self.assertEqual(value, 0.0)
        self.assertIsNone(legal)
        c = b.copy(stack=False)
        c.push_uci("a1a8")                                   # a new position: checkmate, not a draw
        value, _ = batched.leaf_status(c, keys, {batched.position_key(b)})
        self.assertEqual(value, -1.0)


if __name__ == "__main__":
    unittest.main()
