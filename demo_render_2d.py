"""
Демо-режим 2D-отрисовки: рисует то, что реально обнаружено на кадре,
БЕЗ sanity_check/infer_move. Для роборуки (full_pipeline.process_turn/
process_video_stream) эти проверки обязательны и не обходятся — здесь же
цель другая: наглядно показать текущий результат детекции+классификации,
включая её ошибки, а не только "чистые" случаи.

Использование:
    python demo_render_2d.py <видео_или_фото> [--frame N] [--corners x1,y1,x2,y2,x3,y3,x4,y4] [--out out.svg]
corners — top_left,top_right,bottom_right,bottom_left, через запятую.
"""
import argparse
import cv2
import chess
import chess.svg

from full_pipeline import detect_board_state
from board_state import PIECE_TO_SYMBOL


def state_to_board(square_to_piece: dict) -> chess.Board:
    """Строит chess.Board прямо из детекции, без проверки на легальность позиции."""
    board = chess.Board()
    board.clear()
    for square_name, label in square_to_piece.items():
        symbol = PIECE_TO_SYMBOL.get(label)
        if symbol is None:
            continue
        square = chess.parse_square(square_name)
        board.set_piece_at(square, chess.Piece.from_symbol(symbol))
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--corners", required=True, help="x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL)")
    parser.add_argument("--out", default="demo_position.svg")
    args = parser.parse_args()

    vals = [float(v) for v in args.corners.split(",")]
    corners = {
        "top_left": vals[0:2],
        "top_right": vals[2:4],
        "bottom_right": vals[4:6],
        "bottom_left": vals[6:8],
    }

    if args.source.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Не удалось прочитать кадр {args.frame}")
    else:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit("Не удалось прочитать изображение")

    state = detect_board_state(frame, corners)
    print(f"Обнаружено фигур: {len(state)}")
    board = state_to_board(state)

    svg_board = chess.svg.board(board=board, size=480)
    with open(args.out, "w") as f:
        f.write(svg_board)
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
