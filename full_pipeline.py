import json
import cv2
import numpy as np
import chess
import chess.svg
from pathlib import Path
from ultralytics import YOLO
from board_state import PIECE_TO_SYMBOL, infer_move, board_to_state
from stable_detector import StableBoardDetector, sanity_check
from board_corners import calibrate, load_calibration

DETECTOR_PATH = "models/detector.pt"
CLASSIFIER_PATH = "models/classifier.pt"
BOARD_SIZE = 800

detector = YOLO(DETECTOR_PATH)
classifier = YOLO(CLASSIFIER_PATH)

def pixel_to_square(x, y, H):
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)[0][0]
    col = int(transformed[0] // (BOARD_SIZE / 8))
    row = int(transformed[1] // (BOARD_SIZE / 8))
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    cols = "abcdefgh"
    rows = "87654321"
    return f"{cols[col]}{rows[row]}"

def corners_to_homography(corners):
    src_points = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], dtype=np.float32)
    dst_points = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_points, dst_points)

def detect_board_state(frame, corners):
    H = corners_to_homography(corners)

    det_results = detector.predict(frame, conf=0.5, verbose=False)[0]
    square_to_piece = {}

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue

        cls_results = classifier.predict(crop, verbose=False)[0]
        label = cls_results.names[cls_results.probs.top1]

        cx, cy = (x1 + x2) / 2, y2
        square = pixel_to_square(cx, cy, H)
        square_to_piece[square] = label

    return square_to_piece

def process_turn(board: chess.Board, frame, corners):
    curr_state = detect_board_state(frame, corners)
    move, status = infer_move(curr_state, board)

    result = {
        "detected_state": curr_state,
        "move": str(move) if move else None,
        "status": status,
    }

    if status == "valid":
        board.push(move)
        result["message"] = f"Ход {move} принят"
    elif status == "no_change":
        result["message"] = "Изменений на доске нет"
    else:
        result["message"] = "НЕВАЛИДНЫЙ ХОД — верните фигуру на место"

    result["fen"] = board.fen()
    return result, board

def render_board_2d(board: chess.Board, output_path="current_position.svg"):
    svg_board = chess.svg.board(board=board, size=400)
    with open(output_path, "w") as f:
        f.write(svg_board)
    print(f"2D-позиция сохранена: {output_path}")

def run_sequence(image_ids):
    DATAROOT = Path("chessred_data")
    with open(DATAROOT / "annotations.json") as f:
        data = json.load(f)

    images_by_id = {img['id']: img for img in data['images']}
    corners_by_image = {c['image_id']: c['corners'] for c in data['annotations']['corners']}

    board = chess.Board()
    for img_id in image_ids:
        img_info = images_by_id[img_id]
        frame = cv2.imread(str(DATAROOT / img_info['path']))
        corners = corners_by_image[img_id]

        result, board = process_turn(board, frame, corners)
        print(f"image_id={img_id}: статус={result['status']}, ход={result['move']}, {result['message']}")

def process_video_stream(video_path, corners=None, initial_board=None,
                          conf_threshold_frames=3, calibrate_if_missing=True,
                          verbose=True):
    """
    Покадрово детектирует состояние доски, сглаживает через
    StableBoardDetector, прогоняет через sanity_check и определяет ход
    через infer_move().

    corners: если не передан, берётся из camera_calibration.json, а если
    и его нет — калибруется по первому кадру (calibrate_if_missing=True).

    Возвращает (confirmed_moves, board).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    board = initial_board if initial_board else chess.Board()
    stabilizer = StableBoardDetector(history_size=5, min_agreement=conf_threshold_frames)

    if corners is None:
        corners = load_calibration()

    if corners is None:
        if not calibrate_if_missing:
            cap.release()
            raise RuntimeError(
                "Углы доски не заданы и calibrate_if_missing=False. "
                "Передайте corners явно или запустите board_corners.calibrate() заранее."
            )
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Видео пустое — не удалось прочитать первый кадр для калибровки")
        corners = calibrate(first_frame, allow_manual_fallback=False, save=True)
        # Перечитываем видео с начала, чтобы не потерять уже прочитанный первый кадр.
        cap.release()
        cap = cv2.VideoCapture(video_path)

    confirmed_moves = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        raw_state = detect_board_state(frame, corners)
        stable_state = stabilizer.update(raw_state)

        issues = sanity_check(stable_state)
        if issues:
            if verbose:
                print(f"Кадр {frame_num}: проблема детекции — {issues}")
            continue

        move, status = infer_move(stable_state, board)

        if status == "no_change":
            continue
        elif status == "valid":
            board.push(move)
            confirmed_moves.append(str(move))
            if verbose:
                print(f"Кадр {frame_num}: ход принят — {move} (FEN: {board.fen()})")
        else:  # invalid
            if verbose:
                print(f"Кадр {frame_num}: НЕВАЛИДНЫЙ ХОД — верните фигуру на место "
                      f"(увидено: {stable_state})")

    cap.release()
    return confirmed_moves, board

if __name__ == "__main__":
    DATAROOT = Path("chessred_data")
    with open(DATAROOT / "annotations.json") as f:
        data = json.load(f)

    images_by_id = {img['id']: img for img in data['images']}
    corners_by_image = {c['image_id']: c['corners'] for c in data['annotations']['corners']}

    img_id = 0
    img_info = images_by_id[img_id]
    frame = cv2.imread(str(DATAROOT / img_info['path']))
    corners = corners_by_image[img_id]

    board = chess.Board()
    result, board = process_turn(board, frame, corners)

    print("Статус:", result["status"])
    print("Ход:", result["move"])
    print("Сообщение:", result["message"])
    print("FEN:", result["fen"])

    render_board_2d(board)
