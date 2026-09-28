"""recall/precision детектора на ChessReD test: один imgsz против мультимасштаба"""
import json, random
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from detection import detect_boxes

det = YOLO("models/detector.pt")
D = Path("chessred_data"); data = json.load(open(D / "annotations.json"))
imgs = {i["id"]: i for i in data["images"]}
gt = {}
cats = {c["id"] for c in data["categories"] if c["name"] != "empty"}
for p in data["annotations"]["pieces"]:
    if p["category_id"] in cats and "bbox" in p: gt.setdefault(p["image_id"], []).append(p["bbox"])
ids = data["splits"]["chessred2k"]["test"]["image_ids"]
random.seed(1); ids = random.sample(ids, 40)

def iou(a, b):
    ax2, ay2, bx2, by2 = a[2], a[3], b[0] + b[2], b[1] + b[3]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], bx2), min(a[3], by2)
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2]-a[0])*(a[3]-a[1]) + b[2]*b[3] - i
    return i / u

def score(fn, conf):
    tp = fp = fn_ = 0
    for i in ids:
        f = cv2.imread(str(D / imgs[i]["path"])); g = gt.get(i, [])
        b = fn(f, conf); used = set()
        for pb in b:
            best, bj = 0, -1
            for j, gb in enumerate(g):
                if j in used: continue
                v = iou(pb, gb)
                if v > best: best, bj = v, j
            if best >= 0.5: tp += 1; used.add(bj)
            else: fp += 1
        fn_ += len(g) - len(used)
    print(f"  recall {tp/(tp+fn_):.3f} precision {tp/(tp+fp):.3f}  (tp {tp} fp {fp} fn {fn_})")

single = lambda f, c: det.predict(f, conf=c, imgsz=640, verbose=False)[0].boxes.xyxy.cpu().numpy()
multi = lambda f, c: detect_boxes(det, f, conf=c)[0]
for c in (0.4, 0.5):
    print("conf", c, "single 640"); score(single, c)
    print("conf", c, "multi"); score(multi, c)
