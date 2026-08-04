import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

DATAROOT = Path("chessred_data")
DETECTOR_PATH = "models/detector.pt"
CLASSIFIER_PATH = "models/classifier.pt"
BOARD_SIZE = 800

detector = YOLO(DETECTOR_PATH)
classifier = YOLO(CLASSIFIER_PATH)

with open(DATAROOT / "annotations.json") as f:
    data = json.load(f)

images_by_id = {img['id']: img for img in data['images']}
corners_by_image = {c['image_id']: c for c in data['annotations']['corners']}
categories = {c['id']: c['name'] for c in data['categories']}
pieces_by_image = {}
for p in data['annotations']['pieces']:
    pieces_by_image.setdefault(p['image_id'], []).append(p)

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

def test_on_image(img_id):
    img_info = images_by_id[img_id]
    img_path = DATAROOT / img_info['path']
    frame = cv2.imread(str(img_path))

    corners_ann = corners_by_image.get(img_id)
    if corners_ann is None:
        print(f"Нет разметки углов для image_id={img_id}")
        return

    corners = corners_ann['corners']
    src_points = np.array([
        corners['top_left'], corners['top_right'],
        corners['bottom_right'], corners['bottom_left']
    ], dtype=np.float32)
    dst_points = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dst_points)

    # Полный эталон: клетка -> название фигуры (тип+цвет)
    ground_truth = {}
    for p in pieces_by_image.get(img_id, []):
        ground_truth[p['chessboard_position']] = categories[p['category_id']]

    det_results = detector.predict(frame, conf=0.5, verbose=False)[0]
    correct_square = 0
    correct_full = 0
    total = 0

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue

        cls_results = classifier.predict(crop, verbose=False)[0]
        predicted_label = cls_results.names[cls_results.probs.top1]

        cx, cy = (x1 + x2) / 2, y2
        predicted_square = pixel_to_square(cx, cy, H)

        total += 1
        if predicted_square in ground_truth:
            correct_square += 1
            if ground_truth[predicted_square] == predicted_label:
                correct_full += 1

    print(f"image_id={img_id}: клетка верна {correct_square}/{total}, клетка+фигура верна {correct_full}/{total}")

if __name__ == "__main__":
    import sys
    img_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_on_image(img_id)
