"""
Углы доски на кадре для гомографии (pixel -> клетка a1-h8).

Порядок углов ВАЖЕН и идёт по шахматам, а не по картинке:
top_left = угол a8, top_right = h8, bottom_right = h1, bottom_left = a1.
Если снимать с белой стороны, это совпадает с обычным TL/TR/BR/BL на кадре,
если доска повёрнута — нет, и тогда клетки в нотации будут перепутаны.

corners: {"top_left": [x,y], "top_right": [x,y],
          "bottom_right": [x,y], "bottom_left": [x,y]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

CALIBRATION_PATH = Path("camera_calibration.json")
BOARD_SIZE = 800  # как в full_pipeline

_LABELS = ["top_left", "top_right", "bottom_right", "bottom_left"]
_CHESS_NAMES = ["a8", "h8", "h1", "a1"]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """4 точки контура -> [top_left, top_right, bottom_right, bottom_left] по картинке."""
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
    """Ищет 4 угла доски по внешнему контуру. Не путать с findChessboardCorners —
    той нужна пустая доска, здесь фигуры на клетках не мешают.
    Ориентацию шахматную не знает, порядок только по картинке."""
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

    # пробовал ещё текстурную сегментацию (Laplacian), Hough-линии и pose-модель на ChessReD —
    # на захламлённых сценах и других досках всё хуже, чем просто контур (а он тоже так себе)


def detect_corners_manual(frame, window_name="Углы доски по порядку: a8, h8, h1, a1"):
    """Клик по 4 углам доски руками. Нужен дисплей, по SSH без X11 не запустится."""
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    display = frame.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_click)

    while len(points) < 4:
        vis = display.copy()
        for i, pt in enumerate(points):
            cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(vis, _CHESS_NAMES[i], (pt[0] + 8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(points) < 4:
            cv2.putText(vis, f"click {_CHESS_NAMES[len(points)]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(window_name, vis)
        if cv2.waitKey(20) & 0xFF == 27:  # Esc — отмена
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return dict(zip(_LABELS, points))


def corners_from_string(text: str) -> dict:
    """'x1,y1,x2,y2,x3,y3,x4,y4' (a8, h8, h1, a1) -> corners"""
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 8:
        raise ValueError("нужно 8 чисел: x1,y1,x2,y2,x3,y3,x4,y4")
    return {label: vals[2 * i:2 * i + 2] for i, label in enumerate(_LABELS)}


def draw_grid(frame, corners):
    """Рисует сетку 8x8 и названия клеток по углам — глазами проверить калибровку."""
    src = np.float32([corners[k] for k in _LABELS])
    dst = np.float32([[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]])
    Hi = cv2.getPerspectiveTransform(dst, src)
    cell = BOARD_SIZE / 8

    def to_px(x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hi)[0][0]
        return int(p[0]), int(p[1])

    vis = frame.copy()
    thick = max(2, frame.shape[0] // 400)
    for k in range(9):
        cv2.line(vis, to_px(k * cell, 0), to_px(k * cell, BOARD_SIZE), (0, 255, 255), thick)
        cv2.line(vis, to_px(0, k * cell), to_px(BOARD_SIZE, k * cell), (0, 255, 255), thick)
    for row in range(8):
        for col in range(8):
            name = "abcdefgh"[col] + "87654321"[row]
            cx, cy = to_px((col + .5) * cell, (row + .5) * cell)
            cv2.putText(vis, name, (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        frame.shape[0] / 1300, (0, 255, 0), max(1, thick // 2))
    return vis


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
        logger.warning("Автодетект углов не сработал, переключаюсь на ручной выбор")
        corners = detect_corners_manual(frame)

    if corners is None:
        raise RuntimeError(
            "Не удалось определить углы доски автоматически. Задайте их один раз: "
            "python board_corners.py pick <видео> (клик, нужен дисплей) или "
            "python board_corners.py set x1,y1,...,x4,y4 (числами), потом проверьте через show."
        )

    if save:
        save_calibration(corners)
        logger.info(f"Калибровка сохранена: {CALIBRATION_PATH}")

    return corners


def _read_frame(src, idx=0):
    if src.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(src)


USAGE = """python board_corners.py pick <видео|фото> [кадр]   клик по углам a8, h8, h1, a1 (нужен дисплей)
python board_corners.py set x1,y1,x2,y2,x3,y3,x4,y4      задать углы числами (a8, h8, h1, a1)
python board_corners.py show <видео|фото> [кадр]         нарисовать сетку 8x8 -> calibration_check.jpg
python board_corners.py auto <видео|фото> [debug.jpg]    автодетект по контуру (ненадёжно)"""


def main():
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    cmd, arg = sys.argv[1], sys.argv[2]

    if cmd == "set":
        save_calibration(corners_from_string(arg))
        logger.info(f"Сохранено в {CALIBRATION_PATH}, проверьте через show")
        return

    idx = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0
    frame = _read_frame(arg, idx)
    if frame is None:
        logger.error(f"Не удалось прочитать кадр из {arg}")
        sys.exit(1)

    if cmd == "pick":
        corners = detect_corners_manual(frame)
        if corners is None:
            logger.error("Отменено")
            sys.exit(1)
        save_calibration(corners)
        logger.info(f"Сохранено в {CALIBRATION_PATH}")
        cv2.imwrite("calibration_check.jpg", draw_grid(frame, corners))
        logger.info("Сетка для проверки: calibration_check.jpg")

    elif cmd == "show":
        corners = load_calibration()
        if corners is None:
            logger.error(f"Нет {CALIBRATION_PATH}, сначала pick или set")
            sys.exit(1)
        cv2.imwrite("calibration_check.jpg", draw_grid(frame, corners))
        logger.info("Сетка для проверки: calibration_check.jpg")

    elif cmd == "auto":
        debug_out = sys.argv[3] if len(sys.argv) > 3 else "corners_debug.jpg"
        result = detect_corners_auto(frame, debug_path=debug_out)
        if result is None:
            logger.error("Автодетект не нашёл доску, используйте pick или set")
            sys.exit(1)
        print(json.dumps(result, indent=2))

    else:
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
