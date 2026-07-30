import json
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

DATAROOT = Path("chessred_data")
OUTPUT = Path("chessred2k_yolo")

with open(DATAROOT / "annotations.json") as f:
    data = json.load(f)

categories = {c['id']: c['name'] for c in data['categories'] if c['name'] != 'empty'}
images_by_id = {img['id']: img for img in data['images']}
pieces = data['annotations']['pieces']

pieces_by_image = defaultdict(list)
for p in pieces:
    if p['category_id'] in categories:
        pieces_by_image[p['image_id']].append(p)

splits = data['splits']['chessred2k']

for split_name, split_data in splits.items():
    img_dir = OUTPUT / split_name / "images"
    lbl_dir = OUTPUT / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_id in split_data['image_ids']:
        img_info = images_by_id[img_id]
        src_path = DATAROOT / img_info['path']

        if not src_path.exists():
            print(f"WARNING: файл не найден {src_path}")
            continue

        dst_path = img_dir / src_path.name
        shutil.copy(src_path, dst_path)

        with Image.open(src_path) as im:
            img_w, img_h = im.size

        label_lines = []
        for p in pieces_by_image.get(img_id, []):
            x, y, w, h = p['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            label_lines.append(f"{p['category_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = lbl_dir / (src_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

    print(f"{split_name}: обработано {len(split_data['image_ids'])} фото")

names_sorted = [categories[i] for i in sorted(categories.keys())]
yaml_content = f"""train: train/images
val: val/images
test: test/images
nc: {len(names_sorted)}
names: {names_sorted}
"""
(OUTPUT / "data.yaml").write_text(yaml_content)

print("Готово! Датасет в формате YOLO лежит в", OUTPUT)
