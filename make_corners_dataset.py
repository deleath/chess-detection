"""датасет для модели углов доски: YOLO-pose, 1 класс "board", 4 ключевые точки (TL, TR, BR, BL)"""
import json
import os
import shutil
from pathlib import Path

DATAROOT = Path("chessred_data")
OUT = Path("chessred2k_corners")
ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

data = json.load(open(DATAROOT / "annotations.json"))
imgs = {i["id"]: i for i in data["images"]}
corners = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}

outside = 0
for split, sp in data["splits"]["chessred2k"].items():
    img_dir = OUT / split / "images"
    lbl_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for iid in sp["image_ids"]:
        info = imgs[iid]
        src = DATAROOT / info["path"]
        if not src.exists():
            continue
        dst = img_dir / src.name
        if not dst.exists():
            try:
                os.link(src, dst)  # без копирования, места не ест
            except OSError:
                shutil.copy(src, dst)

        w, h = info["width"], info["height"]
        pts = [corners[iid][k] for k in ORDER]
        xs = [p[0] / w for p in pts]
        ys = [p[1] / h for p in pts]
        if min(xs) < 0 or max(xs) > 1 or min(ys) < 0 or max(ys) > 1:
            outside += 1

        # bbox доски + небольшой запас, обрезаем по кадру
        pad = 0.03
        x1, x2 = max(0, min(xs) - pad), min(1, max(xs) + pad)
        y1, y2 = max(0, min(ys) - pad), min(1, max(ys) + pad)
        line = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        for x, y in zip(xs, ys):
            inside = 0 <= x <= 1 and 0 <= y <= 1
            line += [min(max(x, 0), 1), min(max(y, 0), 1), 2 if inside else 1]
        (lbl_dir / (src.stem + ".txt")).write_text(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in line))

    print(split, len(sp["image_ids"]))

(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve()}\n"
    "train: train/images\nval: val/images\ntest: test/images\n"
    "names:\n  0: board\n"
    "kpt_shape: [4, 3]\nflip_idx: [0, 1, 2, 3]\n"
)
print("фото, где угол за кадром:", outside)
print("готово:", OUT)
