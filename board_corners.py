"""
Углы доски на кадре для гомографии (pixel -> клетка a1-h8).
Автодетект по внешнему контуру, ручной клик как fallback.

corners: {"top_left": [x,y], "top_right": [x,y],
          "bottom_right": [x,y], "bottom_left": [x,y]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_PATH = Path("camera_calibration.json")

_LABELS = ["top_left", "top_right", "bottom_right", "bottom_left"]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """4 точки контура -> [top_left, top_right, bottom_right, bottom_left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top_left: минимальная x+y
    ordered[2] = pts[np.argmax(s)]  # bottom_right: максимальная x+y

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]  # top_right: минимальная y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom_left: максимальная y-x

    return ordered


def detect_corners_auto(frame, min_area_ratio=0.15, debug_path=None):
    """
    Ищет 4 угла доски по внешнему контуру (не findChessboardCorners —
    той нужна пустая доска). Возвращает dict или None.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx))

    if not candidates:
        return None

    # берём самый крупный подходящий четырёхугольник
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, best_approx = candidates[0]

    ordered = _order_corners(best_approx)

    if debug_path:
        vis = frame.copy()
        cv2.drawContours(vis, [best_approx], -1, (0, 255, 0), 3)
        for label, pt in zip(["TL", "TR", "BR", "BL"], ordered):
            pt_i = tuple(pt.astype(int))
            cv2.circle(vis, pt_i, 8, (0, 0, 255), -1)
            cv2.putText(vis, label, (pt_i[0] + 10, pt_i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(debug_path), vis)

    return {label: ordered[i].tolist() for i, label in enumerate(_LABELS)}


def detect_corners_manual(frame, window_name="Кликните 4 угла доски: TL, TR, BR, BL, затем любую клавишу"):
    """Клик по 4 углам доски. Нужен дисплей, по SSH без X11 не работает."""
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    display = frame.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    while len(points) < 4:
        vis = display.copy()
        for i, pt in enumerate(points):
            cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(vis, _LABELS[i], (pt[0] + 8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(window_name, vis)
        if cv2.waitKey(20) & 0xFF == 27:  # Esc — отмена калибровки
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return dict(zip(_LABELS, points))


def save_calibration(corners: dict, path: Path = CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(corners, f, indent=2)


def load_calibration(path: Path = CALIBRATION_PATH):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def calibrate(frame, allow_manual_fallback=True, save=True, debug_path=None):
    """Автодетект -> ручной клик как fallback -> сохранить в camera_calibration.json."""
    corners = detect_corners_auto(frame, debug_path=debug_path)

    if corners is None and allow_manual_fallback:
        print("Автодетект углов не сработал — переключаюсь на ручной выбор.")
        corners = detect_corners_manual(frame)

    if corners is None:
        raise RuntimeError(
            "Не удалось определить углы доски ни автоматически, ни вручную. "
            "Проверьте, что доска полностью видна в кадре и достаточно "
            "контрастна на фоне стола, либо передайте corners вручную."
        )

    if save:
        save_calibration(corners)
        print(f"Калибровка сохранена: {CALIBRATION_PATH}")

    return corners


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python board_corners.py <путь_к_видео_или_фото> [debug_output.jpg]")
        sys.exit(1)

    src = sys.argv[1]
    debug_out = sys.argv[2] if len(sys.argv) > 2 else "corners_debug.jpg"

    if src.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("Не удалось прочитать кадр из видео")
            sys.exit(1)
    else:
        frame = cv2.imread(src)
        if frame is None:
            print("Не удалось прочитать изображение")
            sys.exit(1)

    result = detect_corners_auto(frame, debug_path=debug_out)
    if result is None:
        print("Автодетект не нашёл углы доски. Нужна ручная калибровка "
              "(на машине с дисплеем) или другой кадр.")
        sys.exit(1)

    print("Найдены углы:")
    print(json.dumps(result, indent=2))
    print(f"Отладочное изображение с разметкой: {debug_out}")
