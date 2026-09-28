"""leave-one-video-out для классификатора: ChessReD-реплей + размеченные кропы двух видео,
проверка на третьем. Показывает, помогает ли добавление новых доменов на неувиденных.

python loo_classifier.py base                 - метрики без дообучения (baseline)
python loo_classifier.py train chess_test     - дообучить без этого видео и проверить на нём
python loo_classifier.py final                - дообучить на всех размеченных кропах
"""
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from crop_utils import square as _square

BASE = "models_v2_backup/classifier_pre_ft_color.pt"
REPLAY = Path("chessred2k_classifier")
LABELED = Path("harvest/labeled")
VIDEOS = ["chess_test", "chess_wooden_pieces", "chess_occlusion"]
REPLAY_PER_CLASS = 250
DUP = 3  # размеченные кропы повторяем, реплея заметно больше
IMGSZ = int(os.environ.get("IMGSZ", 128))
# PAD: дополнять кроп до квадрата, иначе ultralytics режет центральный квадрат и теряет верх фигуры
PAD = os.environ.get("PAD", "1") == "1"
TAG = f"{'pad' if PAD else 'crop'}{IMGSZ}"


def square(img):
    return _square(img) if PAD else img


def put(src, dst):
    cv2.imwrite(str(dst), square(cv2.imread(str(src))))


CLASSES = sorted(p.name for p in (REPLAY / "train").iterdir())


def labeled(video):
    return [(f, d.name) for d in LABELED.iterdir() for f in d.glob(f"{video}_*.jpg")]


def build(hold):
    out = Path(f"ds_loo_{hold or 'final'}_{TAG}")
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "val"):
        for c in CLASSES:
            (out / split / c).mkdir(parents=True)
    random.seed(0)
    for c in CLASSES:
        files = sorted((REPLAY / "train" / c).glob("*.jpg"))
        for f in random.sample(files, min(REPLAY_PER_CLASS, len(files))):
            put(f, out / "train" / c / f.name)
    for v in VIDEOS:
        for f, c in labeled(v):
            if v == hold:
                put(f, out / "val" / c / f.name)
            else:
                for k in range(DUP):
                    put(f, out / "train" / c / f"{f.stem}_{k}.jpg")
    if hold is None:  # финал: val - кусочек реплея, чтобы обучение вообще что-то мерило
        for c in CLASSES:
            files = sorted((REPLAY / "train" / c).glob("*.jpg"))
            for f in random.sample(files, min(20, len(files))):
                put(f, out / "val" / c / f.name)
    return out


def evaluate(weights, video):
    model = YOLO(weights)
    items = labeled(video)
    res = model.predict([square(cv2.imread(str(f))) for f, _ in items], imgsz=IMGSZ, verbose=False)
    ok = ok_type = ok_color = 0
    confused = Counter()
    for (f, true), r in zip(items, res):
        pred = r.names[r.probs.top1]
        ok += pred == true
        ok_type += pred.split("-")[1] == true.split("-")[1]
        ok_color += pred.split("-")[0] == true.split("-")[0]
        if pred != true:
            confused[(true, pred)] += 1
    n = len(items)
    print(f"{video:22s} n={n:3d}  класс {ok / n:.3f}  тип {ok_type / n:.3f}  цвет {ok_color / n:.3f}")
    for (t, p), k in confused.most_common(6):
        print(f"      {t} -> {p}: {k}")


def train(hold):
    out = build(hold)
    name = f"cls_loo_{hold or 'final'}_{TAG}"
    YOLO(BASE).train(data=str(out), epochs=20, imgsz=IMGSZ, batch=64, lr0=0.0005, optimizer="AdamW",
                     project="runs", name=name, patience=100, workers=4, exist_ok=True)
    return f"runs/classify/runs/{name}/weights/last.pt"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "base":
        for w in (BASE, "models/classifier.pt"):
            print(w)
            for v in VIDEOS:
                evaluate(w, v)
    elif cmd == "train":
        w = train(sys.argv[2])
        evaluate(w, sys.argv[2])
    elif cmd == "final":
        train(None)
