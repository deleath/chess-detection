# Chess Piece Detection Pipeline

Детекция и классификация шахматных фигур на фото/видео. Архитектура: детектор (находит фигуры) + отдельный классификатор (определяет тип и цвет).

## Датасет

Используется подвыборка **ChessReD2K** из [ChessReD](https://github.com/tmasouris/end-to-end-chess-recognition) (1442 train / 330 val / 306 test реальных фото со смартфонов, bbox-разметка фигур).

### Скачать датасет

```bash
git clone https://github.com/tmasouris/end-to-end-chess-recognition.git
cd end-to-end-chess-recognition
sudo apt install python3-tk
pip install pandas ttkthemes
python3 chessred.py --dataroot ../chessred_data --download
```

### Подготовить данные для детектора (1 класс "piece")

```bash
python3 convert_to_yolo_single_class.py
```
Результат: `chessred2k_yolo_singleclass/` (train/val/test в YOLO-формате, все фигуры — класс 0).

### Подготовить данные для классификатора (12 классов: тип + цвет)

```bash
python3 make_classifier_dataset.py
```
Результат: `chessred2k_classifier/` (train/val/test, картинки по папкам-классам).

## Обучение

### Детектор

```bash
yolo detect train \
    data=chessred2k_yolo_singleclass/data.yaml \
    model=yolo11m.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs \
    name=detector
```

### Классификатор

```bash
yolo classify train \
    data=chessred2k_classifier \
    model=yolo11m-cls.pt \
    epochs=30 \
    imgsz=128 \
    batch=32 \
    device=0 \
    project=runs \
    name=classifier
```

После обучения скопируй лучшие веса в `models/`:
```bash
mkdir -p models
cp runs/detect/detector/weights/best.pt models/detector.pt
cp runs/classify/classifier/weights/best.pt models/classifier.pt
```

## Метрики (val / test)

| | val | test |
|---|---|---|
| Детектор mAP50 | 0.995 | 0.995 |
| Детектор mAP50-95 | 0.861 | 0.849 |
| Детектор P / R | 0.997 / 0.999 | 0.991 / 0.997 |
| Классификатор top1 | 0.999 | — |

Проверено на отложенном test-сплите — метрики совпадают с val, признаков утечки данных нет.

## Инференс на видео (полный пайплайн)

```bash
python3 pipeline_predict.py <путь_к_видео> <путь_к_результату>
```
Пример:
```bash
python3 pipeline_predict.py chess_video.mp4 result.mp4
```

Готовые веса ожидаются в `models/detector.pt` и `models/classifier.pt`.

## Требования

```bash
pip install ultralytics roboflow pillow opencv-python
```
Обучалось и тестировалось на: Ubuntu 24.04, Python 3.12, NVIDIA RTX 5090, CUDA 13.0, PyTorch 2.13.0.

## Известные ограничения

- Модель может путать цвет фигур при заметном отличии освещения/тона дерева от тренировочных данных (ChessReD снят в конкретных условиях) — частично решено переходом на архитектуру detect+classify и модель YOLO11m
- Небольшая дёрганность рамок между кадрами видео (нет трекинга объектов между кадрами)
