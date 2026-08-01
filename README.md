# Chess Piece Detection Pipeline

Детекция и классификация шахматных фигур на фото/видео. Архитектура: детектор (находит фигуры) + отдельный классификатор (определяет тип и цвет).

## Датасет

Используется подвыборка **ChessReD2K** из [ChessReD](https://github.com/tmasouris/end-to-end-chess-recognition) (1442 train / 330 val / 306 test реальных фото со смартфонов, bbox-разметка фигур).

### Скачать датасет

```bash
git clone https://github.com/tmasouris/end-to-end-chess-recognition.git
cd end-to-end-chess-recognition
pip install pandas ttkthemes  # tkinter ставится системно: sudo apt install python3-tk
python3 chessred.py --dataroot ../chessred_data --download
```

### Подготовить данные для детектора (1 класс "piece")

```bash
python3 convert_to_yolo_single_class.py
```
Результат: `chessred2k_yolo_singleclass/` (train/val/test с images и labels в YOLO-формате, все фигуры — класс 0).

### Подготовить данные для классификатора (12 классов: тип + цвет)

```bash
python3 make_classifier_dataset.py
```
Результат: `chessred2k_classifier/` (train/val/test, картинки разложены по папкам-классам — формат, который ожидает Ultralytics classify).

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
    name=detector_v3_yolo11m
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
    name=classifier_v2_yolo11m
```

## Метрики (val / test)

| | val | test |
|---|---|---|
| Детектор mAP50 | 0.995 | 0.995 |
| Детектор mAP50-95 | 0.861 | 0.849 |
| Детектор P / R | 0.997 / 0.999 | 0.991 / 0.997 |
| Классификатор top1 | 0.999 | — |

## Инференс на видео (полный пайплайн)

```bash
python3 pipeline_predict.py <путь_к_видео> <путь_к_результату>
```
Пример:
```bash
python3 pipeline_predict.py chess_wooden_pieces.mp4 result.mp4
```

Скрипт: детектор находит рамки фигур → каждая вырезается → классификатор определяет тип+цвет по кропу → итоговое видео с подписями.

## Визуализация пайплайна по шагам (для отладки/демонстрации)

```bash
python3 visualize_pipeline_steps.py <путь_к_кадру.jpg> <имя_примера>
```
Сохраняет: исходное фото, рамки от детектора, вырезанные кропы, финальный результат — в `pipeline_visualization/<имя_примера>/`.

## Требования

```bash
pip install ultralytics roboflow pillow opencv-python
```
Модель обучалась и тестировалась на: Ubuntu 24.04, Python 3.12, NVIDIA RTX 5090, CUDA 13.0, PyTorch 2.13.0.

## Известные ограничения

- Модель путает цвет фигур при заметном отличии освещения/тона дерева от тренировочных данных (ChessReD снят в конкретных условиях)
- Небольшая дёрганность рамок между кадрами видео (нет трекинга объектов между кадрами)
