"""сравнение старых и новых весов.

python compare_models.py heldout OLD_DET NEW_DET OLD_CLS NEW_CLS  - метрики на val/test партиях chessred2k, точность по классам
python compare_models.py video VIDEO OLD_DET OLD_CLS NEW_DET NEW_CLS [n_frames]  - кадры вне датасета: счёт фигур и оверлеи
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def heldout(old_det, new_det, old_cls, new_cls):
    for split in ("val", "test"):
        print(f"\n=== детектор, {split} ===")
        for name, w in (("старый", old_det), ("новый", new_det)):
            m = YOLO(w).val(data="chessred_expanded_det/data.yaml", split=split, imgsz=640, verbose=False, plots=False)
            print(f"  {name}: mAP50 {m.box.map50:.4f}  mAP50-95 {m.box.map:.4f}  P {m.box.mp:.4f}  R {m.box.mr:.4f}")

    for split in ("val", "test"):
        print(f"\n=== классификатор, {split}: точность по классам ===")
        files = sorted(Path(f"chessred_expanded_cls/{split}").glob("*/*.jpg"))
        truth = [f.parent.name for f in files]
        res = {}
        for name, w in (("старый", old_cls), ("новый", new_cls)):
            model = YOLO(w)
            pred = []
            for i in range(0, len(files), 512):
                for r in model.predict([str(f) for f in files[i:i + 512]], verbose=False, imgsz=model.overrides.get("imgsz", 128)):
                    pred.append(r.names[r.probs.top1])
            res[name] = pred
        names = sorted(set(truth))
        print(f"  {'класс':14s} {'n':>6s} {'старый':>8s} {'новый':>8s}")
        for c in names:
            idx = [i for i, t in enumerate(truth) if t == c]
            a = [np.mean([res[n][i] == c for i in idx]) for n in ("старый", "новый")]
            print(f"  {c:14s} {len(idx):6d} {a[0]:8.3f} {a[1]:8.3f}")
        for n in ("старый", "новый"):
            acc = np.mean([p == t for p, t in zip(res[n], truth)])
            col = np.mean([p.split("-")[0] == t.split("-")[0] for p, t in zip(res[n], truth)])
            typ = np.mean([p.split("-")[1] == t.split("-")[1] for p, t in zip(res[n], truth)])
            print(f"  {n}: top1 {acc:.4f} | цвет верен {col:.4f} | тип верен {typ:.4f}")


def count_state(model_det, model_cls, frame):
    r = model_det.predict(frame, conf=0.5, verbose=False)[0]
    labels, boxes = [], []
    for b in r.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, b)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        c = model_cls.predict(crop, verbose=False)[0]
        labels.append(c.names[c.probs.top1])
        boxes.append((x1, y1, x2, y2))
    return labels, boxes


def draw(frame, labels, boxes, title):
    out = frame.copy()
    s = max(1, out.shape[0] // 540)
    for l, (x1, y1, x2, y2) in zip(labels, boxes):
        col = (0, 255, 0) if l.startswith("white") else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 * s)
        cv2.putText(out, l.replace("white-", "W ").replace("black-", "B ").replace("pawn", "p"),
                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, col, s)
    cv2.putText(out, title, (10, 24 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 255, 255), 2 * s)
    return out


def video(path, old_det, old_cls, new_det, new_cls, n_frames=10):
    models = {"старые": (YOLO(old_det), YOLO(old_cls)), "новые": (YOLO(new_det), YOLO(new_cls))}
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 2, n_frames).astype(int)
    stats = {k: {"n": [], "white": [], "black": []} for k in models}
    sheets = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        pair = []
        for k, (d, c) in models.items():
            labels, boxes = count_state(d, c, frame)
            cnt = Counter(l.split("-")[0] for l in labels)
            stats[k]["n"].append(len(labels))
            stats[k]["white"].append(cnt["white"])
            stats[k]["black"].append(cnt["black"])
            tag = {"старые": "OLD", "новые": "NEW"}[k]  # cv2.putText не рисует кириллицу
            pair.append(draw(frame, labels, boxes, f"{tag} | frame {idx} | {len(labels)} pieces (W{cnt['white']}/B{cnt['black']})"))
        sheets.append(np.hstack(pair))
    print(f"\n{Path(path).name}: {len(sheets)} кадров, детект conf>=0.5")
    for k, s in stats.items():
        print(f"  {k}: фигур в кадре в среднем {np.mean(s['n']):.1f} (белых {np.mean(s['white']):.1f}, чёрных {np.mean(s['black']):.1f}), "
              f"макс {max(s['n'])}, кадров с >32 фигурами: {sum(n > 32 for n in s['n'])}")
    out = f"cmp_{Path(path).stem}.jpg"
    keep = [sheets[0], sheets[len(sheets) // 2], sheets[-1]]
    scale = 1600 / keep[0].shape[1]
    cv2.imwrite(out, np.vstack([cv2.resize(k, None, fx=scale, fy=scale) for k in keep]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("оверлеи:", out)


if __name__ == "__main__":
    if sys.argv[1] == "heldout":
        heldout(*sys.argv[2:6])
    elif sys.argv[1] == "video":
        video(*sys.argv[2:7], int(sys.argv[7]) if len(sys.argv) > 7 else 10)
