import cv2
from ultralytics import YOLO
import sys

DETECTOR_PATH = "runs/detect/runs/detector_v3_yolo11m/weights/best.pt"
CLASSIFIER_PATH = "runs/classify/runs/classifier_v2_yolo11m/weights/best.pt"

detector = YOLO(DETECTOR_PATH)
classifier = YOLO(CLASSIFIER_PATH)

def process_video(source_path, output_path):
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Шаг 1: детектор находит фигуры
        det_results = detector.predict(frame, conf=0.5, verbose=False)[0]

        for box in det_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[max(0,y1):y2, max(0,x1):x2]

            if crop.size == 0:
                continue

            # Шаг 2: классификатор определяет тип+цвет
            cls_results = classifier.predict(crop, verbose=False)[0]
            top1_idx = cls_results.probs.top1
            top1_conf = cls_results.probs.top1conf.item()
            label = cls_results.names[top1_idx]

            # Рисуем результат
            color = (0, 255, 0) if 'white' in label else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {top1_conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()
    print(f"Готово: {output_path}")

if __name__ == "__main__":
    source = sys.argv[1]
    output = sys.argv[2]
    process_video(source, output)
