"""Board and move encoding.

The board is always encoded from the side-to-move's perspective: when it is
black's turn the board is mirrored vertically and colors are swapped, so the
network only ever learns "my pieces move up the board". Move indices are
mirrored the same way.

Policy space: from_square * 64 + to_square = 4096 indices. All promotions
share the queen-promotion index; underpromotions are folded into it and
decoding always promotes to a queen.
"""

import chess
import numpy as np

N_PLANES = 17  # 6 own pieces, 6 opponent pieces, 4 castling rights, 1 en passant
POLICY_SIZE = 64 * 64

_PIECE_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

_MATERIAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Returns a (17, 8, 8) uint8 tensor from the mover's perspective."""
    flip = board.turn == chess.BLACK
    x = np.zeros((N_PLANES, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        plane = _PIECE_PLANE[piece.piece_type] + (0 if piece.color == board.turn else 6)
        x[plane, s // 8, s % 8] = 1
    us, them = board.turn, not board.turn
    if board.has_kingside_castling_rights(us):
        x[12, :, :] = 1
    if board.has_queenside_castling_rights(us):
        x[13, :, :] = 1
    if board.has_kingside_castling_rights(them):
        x[14, :, :] = 1
    if board.has_queenside_castling_rights(them):
        x[15, :, :] = 1
    if board.ep_square is not None:
        s = chess.square_mirror(board.ep_square) if flip else board.ep_square
        x[16, s // 8, s % 8] = 1
    return x


def move_to_index(move: chess.Move, flip: bool) -> int:
    f, t = move.from_square, move.to_square
    if flip:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


def legal_move_indices(board: chess.Board):
    """Returns (indices array, index -> canonical Move dict)."""
    flip = board.turn == chess.BLACK
    idx_to_move = {}
    for move in board.legal_moves:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            continue  # folded into the queen-promotion index
        idx_to_move[move_to_index(move, flip)] = move
    return np.fromiter(idx_to_move.keys(), dtype=np.int64), idx_to_move


def material_balance(board: chess.Board) -> int:
    """Material difference from white's perspective, in pawns."""
    score = 0
    for piece in board.piece_map().values():
        v = _MATERIAL[piece.piece_type]
        score += v if piece.color == chess.WHITE else -v
    return score
