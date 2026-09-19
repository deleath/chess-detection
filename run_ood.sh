#!/bin/bash
# сравнение старых и новых весов на видео вне датасета, оба варианта классификатора
cd ~/project_dima/chess-detection
source venv/bin/activate
D=runs/detect/runs/detector_exp/weights/best.pt
for v in chess_test chess_wooden_pieces chess_occlusion chess_medium_shot; do
  for n in 128 192; do
    python compare_models.py video $v.mp4 models/detector.pt models/classifier.pt $D runs/classify/runs/classifier_exp$n/weights/best.pt 10 > ood_${v}_$n.log 2>&1
    mv cmp_$v.jpg cmp_${v}_$n.jpg
  done
done
echo FINISHED > ood_done.flag
