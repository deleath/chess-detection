"""Подготовка кропа фигуры для классификатора."""
import cv2

CLS_IMGSZ = 192


def square(img):
    """Вытянутый кроп -> квадрат с серыми полями. Без этого ultralytics при инференсе режет
    центральный квадрат и теряет верх фигуры (крест короля, митру слона) - тип путался."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
