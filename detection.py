"""Мультимасштабный инференс детектора: фигуры на разных видео очень разного размера
(ChessReD ~ мелкие, вблизи - в разы крупнее), на одном imgsz часть находится, часть нет."""
import torch
from torchvision.ops import box_iou, nms

SCALES = (256, 384, 640, 960)


def detect_boxes(detector, frame, conf=0.4, scales=SCALES, iou=0.5, sure=0.75, min_support=2):
    """xyxy-боксы и conf. Прогоны на нескольких imgsz, дубли схлопнуты NMS. Бокс оставляем,
    если его нашли минимум на min_support масштабах или conf >= sure - иначе на знакомых
    сценах лезут ложные срабатывания от чужих масштабов."""
    boxes, scores, src = [], [], []
    for k, s in enumerate(scales):
        r = detector.predict(frame, conf=conf, imgsz=s, verbose=False)[0]
        boxes.append(r.boxes.xyxy.cpu())
        scores.append(r.boxes.conf.cpu())
        src.append(torch.full((len(r.boxes),), k))
    b, c, src = torch.cat(boxes), torch.cat(scores), torch.cat(src)
    if len(b) == 0:
        return b.numpy(), c.numpy()
    keep = nms(b, c, iou)
    overlap = box_iou(b[keep], b) > iou
    support = torch.stack([(overlap & (src == k)).any(1) for k in range(len(scales))]).sum(0)
    ok = (support >= min_support) | (c[keep] >= sure)
    keep = keep[ok]
    return b[keep].numpy(), c[keep].numpy()
