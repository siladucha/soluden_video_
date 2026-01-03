# video_to_text.py — полный готовый скрипт для твоего репо

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
from tqdm import tqdm
import time
import json
import gc
from PIL import Image

class VideoToText:
    def __init__(self, device='auto', model_size='balanced'):
        if device == 'auto':
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Устройство: {self.device.upper()}")

        configs = {
            'fast': {'yolo': 'yolov8n.pt', 'blip': None, 'sample': 0.3, 'size': 320},
            'balanced': {'yolo': 'yolov8m.pt', 'blip': 'Salesforce/blip2-opt-2.7b', 'sample': 0.5, 'size': 480},
            'accurate': {'yolo': 'yolov8x.pt', 'blip': 'Salesforce/blip2-opt-2.7b', 'sample': 1.0, 'size': 640},
        }
        self.cfg = configs[model_size]

        # YOLO
        self.yolo = YOLO(self.cfg['yolo']).to(self.device)

        # BLIP-2 (если нужен)
        if self.cfg['blip']:
            self.processor = Blip2Processor.from_pretrained(self.cfg['blip'])
            self.blip = Blip2ForConditionalGeneration.from_pretrained(
                self.cfg['blip'], torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.blip = None
            self.processor = None

    def analyze(self, video_path, output=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total <= 0:
            raise ValueError(f"Invalid video properties: fps={fps}, frames={total}")
            
        duration = total / fps

        interval = max(1, int(fps / self.cfg['sample']))
        to_process = total // interval

        print(f"Длительность: {self._fmt(duration)} | Кадров: {total} | Обработаем: {to_process}")

        objects = []
        captions = []
        timeline = []
        frame_idx = 0
        prev_gray = None
        pbar = tqdm(total=to_process, desc="Обработка")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                # Calculate proper aspect ratio to avoid division by zero or negative values
                h, w = frame.shape[:2]
                if w > h:
                    new_w = self.cfg['size']
                    new_h = max(1, int(h * self.cfg['size'] / w))
                else:
                    new_h = self.cfg['size']
                    new_w = max(1, int(w * self.cfg['size'] / h))
                small = cv2.resize(frame, (new_w, new_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and np.mean(cv2.absdiff(gray, prev_gray)) < 10:
                    frame_idx += 1
                    continue
                prev_gray = gray

                res = self.yolo(small, verbose=False)[0]
                frame_objs = []
                if len(res.boxes) > 0:
                    for b in res.boxes:
                        label = self.yolo.names[int(b.cls[0])]
                        conf = float(b.conf[0])
                        if conf > 0.3:
                            frame_objs.append(label)
                            objects.append(label)

                    if self.blip:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
                        ids = self.blip.generate(**inputs, max_new_tokens=30)
                        caption = self.processor.decode(ids[0], skip_special_tokens=True)
                        captions.append(caption)
                        timeline.append({"time": self._fmt(frame_idx / fps), "caption": caption, "objects": frame_objs})

                pbar.update(1)
                if pbar.n % 50 == 0:
                    torch.mps.empty_cache() if self.device == 'mps' else gc.collect()

            frame_idx += 1

        cap.release()
        pbar.close()

        # Хронология
        print("\n=== Подробная хронология сцен ===")
        for entry in timeline:
            objs = Counter(entry['objects']).most_common(3)
            obj_str = ", ".join(f"{o}({c})" for o,c in objs) or "нет"
            print(f"{entry['time']} — {entry['caption']} (объекты: {obj_str})")

        # Сохранение
        result = {
            "summary": f"Видео {self._fmt(duration)} с объектами: {', '.join(set(objects))}",
            "timeline": timeline,
            "objects": dict(Counter(objects))
        }
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Сохранено: {output}")

        return result

    def _fmt(self, sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("-o", "--output")
    parser.add_argument("-m", "--mode", choices=["fast","balanced","accurate"], default="balanced")
    args = parser.parse_args()

    vtt = VideoToText(model_size=args.mode)
    vtt.analyze(args.video, args.output)