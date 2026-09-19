"""восстановление bbox для фото без разметки: углы (pose-модель) -> ориентация -> подгонка под детектор.

Известно для каждого фото: какая фигура на какой клетке (chessboard_position). Не известно: где доска на кадре.
1. pose-модель даёт 4 угла
2. из 4 поворотов выбираем тот, где классификатор на боксах детектора чаще всего согласен с аннотацией
3. подгоняем углы так, чтобы bbox по формуле (bbox_from_squares) совпали с боксами детектора
4. итог: для каждой фигуры бокс детектора, если он совпал с формульным (IoU>0.5), иначе формульный

python reconstruct_bbox.py eval [split]      - проверка на chessred2k (там есть настоящие углы и bbox)
python reconstruct_bbox.py apply N NPARTS    - прогон по фото без углов, можно параллелить по частям
"""
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import bbox_from_squares as bfs

B, CELL, FILES = bfs.B, bfs.CELL, bfs.FILES
POSE_W = "runs/pose/runs/corners_v1/weights/best.pt"

_DST = np.float32([[0, 0], [B, 0], [B, B], [0, B]])
_CEN = np.array([[(i % 8 + .5) * CELL, (i // 8 + .5) * CELL] for i in range(64)])
_QUAD = np.array([[[c * CELL, r * CELL], [(c + 1) * CELL, r * CELL],
                   [(c + 1) * CELL, (r + 1) * CELL], [c * CELL, (r + 1) * CELL]]
                  for r, c in (divmod(i, 8) for i in range(64))])
_RECT = np.concatenate([_CEN, _QUAD.reshape(-1, 2)]).astype(np.float32).reshape(-1, 1, 2)


def squares_px(c):
    """центры и размер s всех 64 клеток на кадре по 4 углам"""
    Hi = cv2.getPerspectiveTransform(_DST, np.float32(c))
    p = cv2.perspectiveTransform(_RECT, Hi).reshape(-1, 2)
    q = p[64:].reshape(64, 4, 2)
    x, y = q[:, :, 0], q[:, :, 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, 1)).sum(1) - (y * np.roll(x, -1, 1)).sum(1))
    return p[:64], np.sqrt(area)


def boxes_from(c, sq, offs):
    cen, s = squares_px(c)
    bx, by, ss = cen[sq, 0], cen[sq, 1], s[sq]
    return np.stack([bx + ss * offs[:, 0], by + ss * offs[:, 1],
                     bx + ss * offs[:, 2], by + ss * offs[:, 3]], 1)


def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def score(c, sq, offs, det):
    if len(det) == 0:
        return 0.0
    return float(iou_matrix(boxes_from(c, sq, offs), det).max(1).mean())


def refine(c0, sq, offs, det, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    best = np.array(c0, dtype=np.float64)
    bs = score(best, sq, offs, det)
    cell = np.mean([np.linalg.norm(best[i] - best[(i + 1) % 4]) for i in range(4)]) / 8
    sigma, fails = 0.25 * cell, 0
    for _ in range(iters):
        cand = best.copy()
        if rng.random() < .5:
            cand[rng.integers(4)] += rng.normal(0, sigma, 2)
        else:
            cand += rng.normal(0, sigma * .5, (4, 2))
        sc = score(cand, sq, offs, det)
        if sc > bs:
            best, bs, fails = cand, sc, 0
        else:
            fails += 1
            if fails >= 15:
                sigma *= .7
                fails = 0
            if sigma < 0.01 * cell:
                break
    return best, bs


def fit_offsets(data, cats, imgs, corners, ids):
    R = bfs.collect(data, cats, imgs, corners)
    R = R[np.isin(R[:, 0].astype(int), list(ids))]
    bx, by, s, gt, cat = R[:, 3], R[:, 4], R[:, 5], R[:, 6:10], R[:, 2].astype(int)
    off = np.stack([(gt[:, 0] - bx) / s, (gt[:, 1] - by) / s, (gt[:, 2] - bx) / s, (gt[:, 3] - by) / s], 1)
    return {c: np.median(off[cat == c], axis=0) for c in set(cat)}


def pieces_by_image(data, cats):
    by = defaultdict(list)
    for p in data["annotations"]["pieces"]:
        if p["category_id"] not in cats:
            continue
        sq = (8 - int(p["chessboard_position"][1])) * 8 + FILES.index(p["chessboard_position"][0])
        gt = None
        if p.get("bbox"):
            x, y, w, h = p["bbox"]
            gt = [x, y, x + w, y + h]
        by[p["image_id"]].append((sq, p["category_id"], gt))
    return by


class Reconstructor:
    def __init__(self, cats, offsets, det_w="models/detector.pt", cls_w="models/classifier.pt", pose_w=POSE_W):
        self.cats, self.offsets = cats, offsets
        self.det, self.cls, self.pose = YOLO(det_w), YOLO(cls_w), YOLO(pose_w)

    def detect(self, img):
        r = self.det.predict(img, conf=0.3, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return boxes, []
        crops = [img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] for x1, y1, x2, y2 in boxes]
        ok = [i for i, c in enumerate(crops) if c.size]
        labels = [""] * len(boxes)
        for i, cr in zip(ok, self.cls.predict([crops[i] for i in ok], verbose=False)):
            labels[i] = cr.names[cr.probs.top1]
        return boxes, labels

    def corners_guess(self, img):
        r = self.pose.predict(img, imgsz=1024, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        return r.keypoints.xy[int(r.boxes.conf.argmax())].cpu().numpy()

    def run(self, img, pieces):
        """pieces: [(sq, cat, gt_or_None)]. -> dict с углами, оценкой и итоговыми боксами"""
        kp = self.corners_guess(img)
        if kp is None:
            return None
        det, labels = self.detect(img)
        if len(det) == 0:
            return None
        sq = np.array([p[0] for p in pieces])
        offs = np.array([self.offsets[p[1]] for p in pieces])
        names = [self.cats[p[1]] for p in pieces]

        best_k, best_agree = 0, -1
        for k in range(4):
            fb = boxes_from(np.roll(kp, k, axis=0), sq, offs)
            m = iou_matrix(fb, det)
            j = m.argmax(1)
            ok = (m[np.arange(len(sq)), j] > 0.3) & np.array([labels[jj] == n for jj, n in zip(j, names)])
            agree = ok.mean()
            if agree > best_agree:
                best_k, best_agree = k, agree
        c, sc = refine(np.roll(kp, best_k, axis=0), sq, offs, det)

        fb = boxes_from(c, sq, offs)
        m = iou_matrix(fb, det)
        j = m.argmax(1)
        use_det = m[np.arange(len(sq)), j] > 0.5
        hybrid = np.where(use_det[:, None], det[j], fb)
        return dict(corners=c, kp_raw=np.roll(kp, best_k, axis=0), shift=best_k, agree=best_agree,
                    score=sc, formula=fb, hybrid=hybrid, use_det=use_det, sq=sq)


def evaluate(split="test", limit=None):
    data, cats, imgs, corners = bfs.load()
    sp = data["splits"]["chessred2k"]
    offsets = fit_offsets(data, cats, imgs, corners, set(sp["train"]["image_ids"]))
    by = pieces_by_image(data, cats)
    rec = Reconstructor(cats, offsets)

    ids = sp[split]["image_ids"][:limit]
    rows, fails, wrong_shift = [], 0, 0
    for n, iid in enumerate(ids):
        img = cv2.imread(f"chessred_data/{imgs[iid]['path']}")
        pcs = by[iid]
        out = rec.run(img, pcs)
        if out is None:
            fails += 1
            continue
        gtc = np.array([corners[iid][k] for k in ("top_left", "top_right", "bottom_right", "bottom_left")])
        cell = np.mean([np.linalg.norm(gtc[i] - gtc[(i + 1) % 4]) for i in range(4)]) / 8
        # какой поворот на самом деле правильный (по настоящим углам)
        true_k = int(np.argmin([np.linalg.norm(np.roll(rec.corners_guess(img), k, axis=0) - gtc, axis=1).mean() for k in range(4)]))
        wrong_shift += int(out["shift"] != true_k)
        gt = np.array([p[2] for p in pcs])
        rows.append(dict(
            err_raw=np.linalg.norm(out["kp_raw"] - gtc, axis=1).mean() / cell,
            err_fin=np.linalg.norm(out["corners"] - gtc, axis=1).mean() / cell,
            score=out["score"], agree=out["agree"],
            iou_f=np.diag(iou_matrix(out["formula"], gt)), iou_h=np.diag(iou_matrix(out["hybrid"], gt)),
            det_share=out["use_det"].mean()))
        if n % 50 == 0:
            print(n, "/", len(ids), flush=True)

    print(f"\nфото: {len(rows)} обработано, {fails} провалов (нет углов/боксов), неверная ориентация: {wrong_shift}")
    f = lambda k: np.array([r[k] for r in rows])
    print(f"ошибка углов, в клетках: до подгонки {f('err_raw').mean():.3f}, после {f('err_fin').mean():.3f} "
          f"(медиана {np.median(f('err_fin')):.3f})")
    for name in ("iou_f", "iou_h"):
        v = np.concatenate([r[name] for r in rows])
        label = "формула" if name == "iou_f" else "гибрид (бокс детектора там, где совпал)"
        print(f"IoU {label}: среднее {v.mean():.3f}, медиана {np.median(v):.3f}, "
              f">0.5: {(v > .5).mean():.1%}, >0.7: {(v > .7).mean():.1%}")
    print(f"доля боксов от детектора в гибриде: {f('det_share').mean():.1%}")

    print("\nоценка подгонки как фильтр (score -> реальный IoU гибрида):")
    sc = f("score")
    ih = np.array([r["iou_h"].mean() for r in rows])
    for t in (0.0, 0.5, 0.6, 0.7, 0.8):
        m = sc >= t
        if m.any():
            print(f"  score>={t}: проходит {m.mean():.1%} фото, средний IoU {ih[m].mean():.3f}")


def apply(part=0, nparts=1):
    """фото без углов -> reconstructed_partN.jsonl (по строке на фото, можно докачивать)"""
    data, cats, imgs, corners = bfs.load()
    offsets = fit_offsets(data, cats, imgs, corners, set(corners))
    by = pieces_by_image(data, cats)
    rec = Reconstructor(cats, offsets)

    todo = sorted(i for i in imgs if i not in corners)[part::nparts]
    path = f"reconstructed_part{part}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path)}
    print(f"часть {part}/{nparts}: {len(todo)} фото, уже готово {len(done)}", flush=True)

    with open(path, "a") as f:
        for n, iid in enumerate(todo):
            if iid in done:
                continue
            rec_line = {"id": iid, "ok": False}
            try:
                img = cv2.imread(f"chessred_data/{imgs[iid]['path']}")
                pcs = by[iid]
                out = rec.run(img, pcs)
                if out is not None:
                    rec_line = {
                        "id": iid, "ok": True, "score": out["score"], "agree": out["agree"],
                        "shift": out["shift"], "corners": out["corners"].tolist(),
                        "items": [{"sq": int(pcs[i][0]), "cat": int(pcs[i][1]),
                                   "box": [round(float(v), 1) for v in out["hybrid"][i]],
                                   "src": "det" if out["use_det"][i] else "formula"}
                                  for i in range(len(pcs))]}
            except Exception as e:  # одно битое фото не должно ронять весь прогон
                rec_line["error"] = str(e)[:200]
            f.write(json.dumps(rec_line) + "\n")
            f.flush()
            if n % 200 == 0:
                print(part, n, "/", len(todo), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "test",
                 int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == "apply":
        apply(int(sys.argv[2]), int(sys.argv[3]))
