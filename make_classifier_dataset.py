import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

DATAROOT = Path("chessred_data")
OUTPUT = Path("chessred2k_classifier")

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
    for cat_name in categories.values():
        (OUTPUT / split_name / cat_name).mkdir(parents=True, exist_ok=True)

    for img_id in split_data['image_ids']:
        img_info = images_by_id[img_id]
        src_path = DATAROOT / img_info['path']
        if not src_path.exists():
            continue

        with Image.open(src_path) as im:
            for p in pieces_by_image.get(img_id, []):
                x, y, w, h = p['bbox']
                pad_w, pad_h = w * 0.05, h * 0.05
                crop = im.crop((
                    max(0, x - pad_w),
                    max(0, y - pad_h),
                    x + w + pad_w,
                    y + h + pad_h
                ))
                cat_name = categories[p['category_id']]
                crop_path = OUTPUT / split_name / cat_name / f"{p['id']}.jpg"
                crop.convert("RGB").save(crop_path)

    print(f"{split_name}: готово")

print("Готово:", OUTPUT)
