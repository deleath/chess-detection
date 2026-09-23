"""достаёт кропы фигур с проблемных видео старым детектором, метит их текущим лучшим
классификатором (тип берём как есть, цвет доразмечаю глазами), складывает в контакт-листы.

python harvest_crops.py harvest   - собрать кропы + контакт-листы
python harvest_crops.py apply CORRECTIONS.json  - применить разметку цвета, разложить по папкам
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEOS = ["chess_test", "chess_wooden_pieces", "chess_occlusion"]
FRAMES_PER_VIDEO = 14
DET_W = "models/detector.pt"
CLS_W = "runs/classify/runs/classifier_exp192/weights/best.pt"
OUT = Path("harvest")
SHEET_COLS, SHEET_ROWS = 6, 6
CELL = 140


def harvest():
    det, cls = YOLO(DET_W), YOLO(CLS_W)
    OUT.mkdir(exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    meta = []
    for video in VIDEOS:
        cap = cv2.VideoCapture(video + ".mp4")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in np.linspace(0, n - 2, FRAMES_PER_VIDEO).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            r = det.predict(frame, conf=0.4, verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 8:
                    continue
                c = cls.predict(crop, verbose=False)[0]
                label = c.names[c.probs.top1]
                cid = len(meta)
                cv2.imwrite(str(OUT / "crops" / f"{cid}.jpg"), crop)
                meta.append({"id": cid, "video": video, "frame": int(fi), "pred": label})
        print(video, "готово, кропов всего:", len(meta), flush=True)

    json.dump(meta, open(OUT / "meta.json", "w"))
    print("всего кропов:", len(meta))

    per_sheet = SHEET_COLS * SHEET_ROWS
    for s in range(0, len(meta), per_sheet):
        batch = meta[s:s + per_sheet]
        sheet = np.full((SHEET_ROWS * CELL, SHEET_COLS * CELL, 3), 40, np.uint8)
        for k, m in enumerate(batch):
            r, cc = divmod(k, SHEET_COLS)
            crop = cv2.imread(str(OUT / "crops" / f"{m['id']}.jpg"))
            ch, cw = crop.shape[:2]
            scale = min((CELL - 24) / ch, (CELL - 10) / cw)
            crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
            y0 = r * CELL + 20
            x0 = cc * CELL + (CELL - crop.shape[1]) // 2
            sheet[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
            col = (0, 255, 0) if m["pred"].startswith("white") else (0, 120, 255)
            cv2.putText(sheet, f"#{m['id']} {m['pred'][:1].upper()}", (cc * CELL + 3, r * CELL + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.imwrite(str(OUT / f"sheet_{s // per_sheet}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("листов:", (len(meta) + per_sheet - 1) // per_sheet)


def apply(corr_path):
    """corr_path: json {"black": [id,...], "white": [id,...], "skip": [id,...]} -
    только те, что расходятся с pred или требуют подтверждения; остальные берём как pred."""
    meta = {m["id"]: m for m in json.load(open(OUT / "meta.json"))}
    corr = json.load(open(corr_path))
    black_ids, white_ids, skip_ids = set(corr.get("black", [])), set(corr.get("white", [])), set(corr.get("skip", []))

    dest = OUT / "labeled"
    n_by_class = {}
    for cid, m in meta.items():
        if cid in skip_ids:
            continue
        typ = m["pred"].split("-", 1)[1]
        if cid in black_ids:
            color = "black"
        elif cid in white_ids:
            color = "white"
        else:
            color = m["pred"].split("-", 1)[0]  # доверяем модели, если не поправили
        cls_name = f"{color}-{typ}"
        d = dest / cls_name
        d.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(OUT / "crops" / f"{cid}.jpg"))
        cv2.imwrite(str(d / f"{m['video']}_{cid}.jpg"), img)
        n_by_class[cls_name] = n_by_class.get(cls_name, 0) + 1
    print("размечено:", sum(n_by_class.values()), "по классам:")
    for k, v in sorted(n_by_class.items()):
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    if sys.argv[1] == "harvest":
        harvest()
    elif sys.argv[1] == "apply":
        apply(sys.argv[2])
