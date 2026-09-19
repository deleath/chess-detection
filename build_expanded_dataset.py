"""расширенные датасеты для детектора и классификатора.

train = chessred2k train (настоящие bbox) + восстановленные фото (reconstructed_part*.jsonl, score >= порога)
val/test = chessred2k val/test как раньше, там партии, которых нет в train, метрики сравнимы со старыми

python build_expanded_dataset.py [--thr 0.75]
результат: chessred_expanded_det/ (YOLO, 1 класс) и chessred_expanded_cls/ (папки-классы)
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DATAROOT = Path("chessred_data")
DET_OUT = Path("chessred_expanded_det")
CLS_OUT = Path("chessred_expanded_cls")
FILES = "abcdefgh"


def load():
    data = json.load(open(DATAROOT / "annotations.json"))
    cats = {c["id"]: c["name"] for c in data["categories"] if c["name"] != "empty"}
    imgs = {i["id"]: i for i in data["images"]}
    gt = {}  # image_id -> [(cat_id, xyxy, piece_id)], только настоящие bbox
    for p in data["annotations"]["pieces"]:
        if p["category_id"] in cats and p.get("bbox"):
            x, y, w, h = p["bbox"]
            gt.setdefault(p["image_id"], []).append((p["category_id"], [x, y, x + w, y + h], f"p{p['id']}"))
    return data, cats, imgs, gt


def load_recon(thr):
    out = {}
    for f in glob.glob("reconstructed_part*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["ok"] and r["score"] >= thr:
                out[r["id"]] = [(it["cat"], it["box"], f"r{r['id']}_{it['sq']}") for it in r["items"]]
    return out


def link(src, dst):
    if not dst.exists():
        try:
            os.link(src, dst)  # без копирования
        except OSError:
            shutil.copy(src, dst)


def build_det(splits, imgs):
    for split, items in splits.items():
        (DET_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (DET_OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for iid, boxes in items.items():
            info = imgs[iid]
            src = DATAROOT / info["path"]
            link(src, DET_OUT / split / "images" / src.name)
            w, h = info["width"], info["height"]
            lines = []
            for _, (x1, y1, x2, y2), _ in boxes:
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (DET_OUT / split / "labels" / (src.stem + ".txt")).write_text("\n".join(lines))
            n_boxes += len(lines)
        print(f"det {split}: {len(items)} фото, {n_boxes} боксов", flush=True)
    (DET_OUT / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['piece']\n")


def crop_worker(task):
    split, path, jobs = task
    n = 0
    with Image.open(path) as im:
        W, H = im.size
        for cat_name, (x1, y1, x2, y2), name in jobs:
            pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05  # запас 5%, как в старом make_classifier_dataset
            box = (max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            im.crop(box).convert("RGB").save(CLS_OUT / split / cat_name / f"{name}.jpg")
            n += 1
    return split, n


def build_cls(splits, imgs, cats):
    for split in splits:
        for name in cats.values():
            (CLS_OUT / split / name).mkdir(parents=True, exist_ok=True)
    tasks = [(split, str(DATAROOT / imgs[iid]["path"]), [(cats[c], b, n) for c, b, n in boxes])
             for split, items in splits.items() for iid, boxes in items.items()]
    counts = Counter()
    with Pool(24) as pool:
        for i, (split, n) in enumerate(pool.imap_unordered(crop_worker, tasks, chunksize=8)):
            counts[split] += n
            if i % 1000 == 0:
                print(f"cls: {i}/{len(tasks)} фото", flush=True)
    print("cls кропов:", dict(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.75)
    args = ap.parse_args()

    data, cats, imgs, gt = load()
    recon = load_recon(args.thr)
    sp = data["splits"]["chessred2k"]
    splits = {
        "train": {**{i: gt[i] for i in sp["train"]["image_ids"] if i in gt}, **recon},
        "val": {i: gt[i] for i in sp["val"]["image_ids"] if i in gt},
        "test": {i: gt[i] for i in sp["test"]["image_ids"] if i in gt},
    }
    print(f"train: {len(sp['train']['image_ids'])} настоящих + {len(recon)} восстановленных (score >= {args.thr})")
    build_det(splits, imgs)
    build_cls(splits, imgs, cats)
    print("готово")


if __name__ == "__main__":
    main()
