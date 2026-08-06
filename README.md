# chess-detection

Пайплайн детекции и классификации шахматных фигур на базе YOLO11.

## Датасет

```bash
git clone https://github.com/tmasouris/end-to-end-chess-recognition.git
cd end-to-end-chess-recognition
sudo apt install python3-tk
pip install pandas ttkthemes
python3 chessred.py --dataroot ../chessred_data --download
```

```bash
python3 convert_to_yolo_single_class.py   # -> chessred2k_yolo_singleclass/
python3 make_classifier_dataset.py        # -> chessred2k_classifier/
```

## Обучение

```bash
yolo detect train \
    data=chessred2k_yolo_singleclass/data.yaml \
    model=yolo11m.pt epochs=50 imgsz=640 batch=16 device=0 \
    project=runs name=detector

yolo classify train \
    data=chessred2k_classifier \
    model=yolo11m-cls.pt epochs=30 imgsz=128 batch=32 device=0 \
    project=runs name=classifier
```

```bash
mkdir -p models
cp runs/detect/runs/detector/weights/best.pt models/detector.pt
cp runs/classify/runs/classifier/weights/best.pt models/classifier.pt
```

(ultralytics с `project=runs` сам добавляет `detect/`/`classify/` перед именем проекта — отсюда двойное `runs`, это не опечатка)

## Готовые веса

Без обучения — веса лежат в [релизе v1.0-models](https://github.com/deleath/chess-detection/releases/tag/v1.0-models):

```bash
mkdir -p models
curl -L -o models/detector.pt https://github.com/deleath/chess-detection/releases/download/v1.0-models/detector.pt
curl -L -o models/classifier.pt https://github.com/deleath/chess-detection/releases/download/v1.0-models/classifier.pt
```

## Инференс

```bash
python3 pipeline_predict.py <видео> <результат>
```

Полный цикл — с нотацией, валидацией хода и сглаживанием детекции:

```python
from full_pipeline import process_video_stream
moves, board = process_video_stream("game.mp4")
```

## Требования

```bash
pip install ultralytics numpy pillow opencv-python chess
```
