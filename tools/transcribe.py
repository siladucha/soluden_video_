import cv2
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# YOLO для выборки интересных кадров
yolo = YOLO('yolov8n.pt').to(device)

# BLIP-2 для captioning (скачается ~1–2 GB в первый раз)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

cap = cv2.VideoCapture('istockphoto-483534858-640_adpp_is.mp4')
captions = []
frame_count = 0
target_fps = 2  # берём 2 кадра в секунду для описания

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / target_fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        # Сначала проверяем, есть ли объекты (YOLO)
        results = yolo(frame, verbose=False)
        if len(results[0].boxes) > 0:  # только интересные кадры
            inputs = processor(images=frame, return_tensors="pt").to(device)
            generated_ids = caption_model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            captions.append(caption)
            print(f"Кадр {frame_count}: {caption}")
    frame_count += 1

cap.release()

# Итоговое описание
print("\n=== Полные описания ключевых кадров ===")
for c in captions:
    print("- ", c)

print("\n=== Общее описание видео ===")
print(" ".join(set(captions)))  # или вручную объедини