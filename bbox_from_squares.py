"""bbox фигуры из клетки + углов доски. Проверка на chessred2k, там есть настоящие bbox.

Идея: основание фигуры = центр клетки (обратная гомография), размер клетки на кадре = s.
bbox считаем как смещение от основания в единицах s, смещения берём медианой по классу.
Валидация по партиям (5 фолдов), чтобы не было утечки между фото одной партии.
"""
import json
from collections import defaultdict

import cv2
import numpy as np

DATAROOT = "chessred_data"
B = 800
CELL = B / 8
FILES = "abcdefgh"


def load():
    data = json.load(open(f"{DATAROOT}/annotations.json"))
    cats = {c["id"]: c["name"] for c in data["categories"] if c["name"] != "empty"}
    imgs = {i["id"]: i for i in data["images"]}
    corners = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}
    return data, cats, imgs, corners


def board_to_px(corners):
    src = np.float32([corners["top_left"], corners["top_right"],
                      corners["bottom_right"], corners["bottom_left"]])
    dst = np.float32([[0, 0], [B, 0], [B, B], [0, B]])
    return cv2.getPerspectiveTransform(dst, src)


def to_px(Hi, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hi).reshape(-1, 2)


def cell_geom(Hi, square):
    """основание фигуры на кадре (центр клетки) и размер клетки s (корень из площади)"""
    col = FILES.index(square[0])
    row = 8 - int(square[1])
    center = to_px(Hi, [[(col + .5) * CELL, (row + .5) * CELL]])[0]
    quad = to_px(Hi, [[col * CELL, row * CELL], [(col + 1) * CELL, row * CELL],
                      [(col + 1) * CELL, (row + 1) * CELL], [col * CELL, (row + 1) * CELL]])
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return center, float(np.sqrt(area))


def iou(a, b):
    ix1, iy1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix2, iy2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def collect(data, cats, imgs, corners):
    hi_cache = {}
    rows = []
    for p in data["annotations"]["pieces"]:
        iid = p["image_id"]
        if iid not in corners or not p.get("bbox") or p["category_id"] not in cats:
            continue
        if iid not in hi_cache:
            hi_cache[iid] = board_to_px(corners[iid])
        base, s = cell_geom(hi_cache[iid], p["chessboard_position"])
        x, y, w, h = p["bbox"]
        rows.append((iid, imgs[iid]["game_id"], p["category_id"], base[0], base[1], s,
                     x, y, x + w, y + h))
    return np.array(rows, dtype=np.float64)


def main():
    data, cats, imgs, corners = load()
    R = collect(data, cats, imgs, corners)
    print("фигур с bbox и corners:", len(R))

    game, cat = R[:, 1].astype(int), R[:, 2].astype(int)
    bx, by, s = R[:, 3], R[:, 4], R[:, 5]
    gt = R[:, 6:10]
    off = np.stack([(gt[:, 0] - bx) / s, (gt[:, 1] - by) / s,
                    (gt[:, 2] - bx) / s, (gt[:, 3] - by) / s], axis=1)

    games = sorted(set(game))
    fold_of = {g: i % 5 for i, g in enumerate(games)}
    fold = np.array([fold_of[g] for g in game])

    pred = np.zeros_like(gt)
    for f in range(5):
        tr, te = fold != f, fold == f
        for c in set(cat[te]):
            m = np.median(off[tr & (cat == c)], axis=0)
            idx = te & (cat == c)
            pred[idx] = np.stack([bx[idx] + s[idx] * m[0], by[idx] + s[idx] * m[1],
                                  bx[idx] + s[idx] * m[2], by[idx] + s[idx] * m[3]], axis=1)

    # для сравнения: тупой bbox размером с клетку вокруг основания
    naive = np.stack([bx - s / 2, by - s / 2, bx + s / 2, by + s / 2], axis=1)

    for name, boxes in (("формула", pred), ("bbox=клетка", naive)):
        v = iou(boxes, gt)
        print(f"{name}: IoU среднее {v.mean():.3f}, медиана {np.median(v):.3f}, "
              f">0.5: {(v > .5).mean():.1%}, >0.7: {(v > .7).mean():.1%}")

    v = iou(pred, gt)
    print("\nпо классам (формула):")
    for c in sorted(set(cat)):
        m = cat == c
        print(f"  {cats[c]:14s} n={m.sum():5d}  IoU {v[m].mean():.3f}")

    print("\nпо партиям (формула), худшие 5:")
    per_game = sorted(((v[game == g].mean(), g) for g in games))
    for val, g in per_game[:5]:
        print(f"  game {g}: {val:.3f}")

    print("\nпо камерам:")
    cam = np.array([imgs[int(i)]["camera"] for i in R[:, 0]])
    for c in sorted(set(cam)):
        print(f"  {c}: {v[cam == c].mean():.3f}")


if __name__ == "__main__":
    main()
