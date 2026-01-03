import cv2
from ultralytics import YOLO
import torch

# Определяем устройство: используем MPS (GPU Apple Silicon), если доступно
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Используем устройство: {device.upper()}")

# Загружаем лёгкую и быструю модель YOLOv8 (в первый раз скачает ~6 МБ)
model = YOLO('yolov8n.pt')  # можно заменить на yolov8s.pt для большей точности
model.to(device)

# Путь к твоему тестовому видео из репозитория (замени, если нужно)
# В твоём репо есть sample_video1.mp4 и sample_video2.mp4
video_path = 'sample_video1.mp4'  # или 'sample_video2.mp4' или 0 для веб-камеры
video_path = '/Volumes/2SSD/soluden_video_/istockphoto-483534858-640_adpp_is.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео. Проверь путь к файлу.")
    exit()

print("Нажми 'q' для выхода из окна.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Видео закончилось или ошибка чтения кадра.")
        break

    # Делаем детекцию на выбранном устройстве
    # conf=0.5 — минимальная уверенность, можно понизить до 0.3 если нужно больше объектов
    results = model(frame, conf=0.5, device=device, verbose=False)

    # annotated_frame уже содержит нарисованные боксы + текстовые метки классов
    annotated_frame = results[0].plot()

    # Показываем кадр с детекцией и подписями (например: "person 0.92", "car 0.87")
    cv2.imshow('YOLOv8 на Apple Silicon M4 (MPS)', annotated_frame)

    # Опционально: выводим в консоль найденные объекты
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            print(f"Обнаружен: {label} (уверенность: {conf:.2f})")

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

print("Готово!")