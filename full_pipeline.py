import json
import cv2
import numpy as np
import chess
import chess.svg
from pathlib import Path
from ultralytics import YOLO
from board_state import PIECE_TO_SYMBOL, infer_move, board_to_state

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

def detect_board_state(frame, corners):
    src_points = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], dtype=np.float32)
    dst_points = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dst_points)

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

from stable_detector import StableBoardDetector, sanity_check

def process_video_stream(video_path, initial_board=None, conf_threshold_frames=3):
    """
    Обрабатывает видео целиком: покадрово детектирует доску, сглаживает
    через голосование, и при каждом стабильном изменении пытается
    определить и провалидировать ход.
    """
    cap = cv2.VideoCapture(video_path)
    board = initial_board if initial_board else chess.Board()
    stabilizer = StableBoardDetector(history_size=5, min_agreement=conf_threshold_frames)

    # ВАЖНО: для реального видео понадобятся corners с этого же видео,
    # а не из датасета — пока для демонстрации нужен статичный вызов
    # с заранее известными corners (например, откалиброванными вручную)

    last_confirmed_state = board_to_state(board)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # detect_board_state здесь потребует corners — заглушка,
        # реальный вызов будет после калибровки камеры роборуки
        # raw_state = detect_board_state(frame, corners)
        # stable_state = stabilizer.update(raw_state)

        # issues = sanity_check(stable_state)
        # if issues:
        #     print(f"Кадр {frame_num}: проблема детекции — {issues}")
        #     continue

        # if stable_state != last_confirmed_state:
        #     move, status = infer_move(stable_state, board)
        #     ...

        pass  # заглушка до появления реальной камеры/corners

    cap.release()
