import chess

PIECE_TO_SYMBOL = {
    "white-pawn": "P", "white-rook": "R", "white-knight": "N",
    "white-bishop": "B", "white-queen": "Q", "white-king": "K",
    "black-pawn": "p", "black-rook": "r", "black-knight": "n",
    "black-bishop": "b", "black-queen": "q", "black-king": "k",
}

def board_to_state(board: chess.Board) -> dict:
    """Переводит chess.Board в {"e2": "white-pawn", ...}"""
    state = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            color = "white" if piece.color == chess.WHITE else "black"
            piece_type = chess.piece_name(piece.piece_type)
            state[square_name] = f"{color}-{piece_type}"
    return state

def infer_move(curr_state: dict, board: chess.Board):
    """
    Возвращает (move, status), где status один из:
    "no_change" — позиция не изменилась, хода не было
    "valid"     — найден легальный ход, объясняющий изменение
    "invalid"   — изменение не соответствует ни одному легальному ходу
    """
    if curr_state == board_to_state(board):
        return None, "no_change"

    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        if board_to_state(board_copy) == curr_state:
            return move, "valid"

    return None, "invalid"


if __name__ == "__main__":
    board = chess.Board()

    # Тест 1: позиция не изменилась
    move, status = infer_move(board_to_state(board), board)
    print(f"Тест 1 (без изменений): move={move}, status={status}")

    # Тест 2: валидный ход e2-e4
    after_e4 = dict(board_to_state(board))
    del after_e4["e2"]
    after_e4["e4"] = "white-pawn"
    move, status = infer_move(after_e4, board)
    print(f"Тест 2 (e2-e4): move={move}, status={status}")

    # Тест 3: невалидный "прыжок" коня
    invalid = dict(board_to_state(board))
    del invalid["b1"]
    invalid["b5"] = "white-knight"
    move, status = infer_move(invalid, board)
    print(f"Тест 3 (невалидный): move={move}, status={status}")
