# Video Analysis System

## 📌 Overview
This project is an **Demo AI-powered video analysis system** that combines **object detection** and **scene captioning** to automatically generate detailed descriptions of video content. It uses **YOLO** for object detection and **BLIP-2** for natural language scene descriptions.

## 🚀 Features
- **🎯 Object Detection** – Uses YOLOv8 models to identify and track objects in video frames
- **📝 Scene Captioning** – Generates natural language descriptions using BLIP-2 vision-language model
- **⏱️ Timeline Analysis** – Creates detailed chronological breakdown of video scenes
- **📊 Object Statistics** – Counts and analyzes object frequency throughout the video
- **🔧 Configurable Modes** – Three performance modes: fast, balanced, and accurate
- **� FJSON Export** – Saves analysis results in structured JSON format
- **�️ Smuart Processing** – Automatic device detection (MPS/CPU) and memory optimization

## 🏗️ System Architecture
The system processes videos through a pipeline approach:

1️⃣ **Frame Sampling** – Intelligently samples frames based on motion detection
2️⃣ **Object Detection** – YOLOv8 identifies objects with confidence scoring
3️⃣ **Scene Understanding** – BLIP-2 generates contextual descriptions
4️⃣ **Timeline Generation** – Creates chronological analysis with timestamps
5️⃣ **Statistics & Export** – Aggregates data and exports results

## 🛠️ Installation & Setup
### **Prerequisites**
- **Python 3.8+**
- **PyTorch** with MPS support (for Apple Silicon) or CUDA (for NVIDIA GPUs)

### **Install Required Packages**
```bash
pip install -r requirements.txt
```

### **Dependencies**
- `ultralytics` - YOLOv8 object detection
- `transformers` - BLIP-2 vision-language model
- `opencv-python` - Video processing
- `torch` - Deep learning framework
- `pillow` - Image processing
- `tqdm` - Progress bars
- `numpy` - Numerical computing

## 🚀 Usage

### **Basic Usage**
```bash
python video_to_text.py path/to/video.mp4
```

### **With Output File**
```bash
python video_to_text.py path/to/video.mp4 -o analysis.json
```

### **Performance Modes**
```bash
# Fast mode - Quick processing, basic object detection only
python video_to_text.py video.mp4 -m fast

# Balanced mode - Good balance of speed and accuracy (default)
python video_to_text.py video.mp4 -m balanced

# Accurate mode - Maximum accuracy, slower processing
python video_to_text.py video.mp4 -m accurate
```

### **Handling Files with Spaces**
```bash
python video_to_text.py "path/to/My Video File.mp4" -m balanced
```

## ⚙️ Configuration Modes

| Mode | YOLO Model | BLIP-2 | Sample Rate | Resolution | Use Case |
|------|------------|--------|-------------|------------|----------|
| **Fast** | YOLOv8n | None | 30% | 320px | Quick object detection |
| **Balanced** | YOLOv8m | BLIP-2 2.7B | 50% | 480px | General purpose analysis |
| **Accurate** | YOLOv8x | BLIP-2 2.7B | 100% | 640px | Detailed analysis |

## 📊 Output Format
The system generates detailed analysis including:

### **Console Output**
```
Длительность: 02:15 | Кадров: 3240 | Обработаем: 67
=== Подробная хронология сцен ===
00:05 — A person walking in a park with trees in the background (объекты: person(1), tree(3))
00:12 — A car driving on a street with buildings visible (объекты: car(1), building(2))
```

### **JSON Export Structure**
```json
{
  "summary": "Видео 02:15 с объектами: person, car, tree, building",
  "timeline": [
    {
      "time": "00:05",
      "caption": "A person walking in a park with trees in the background",
      "objects": ["person", "tree", "tree", "tree"]
    }
  ],
  "objects": {
    "person": 15,
    "car": 8,
    "tree": 45,
    "building": 12
  }
}
```

## 🔧 Technical Details

### **Smart Frame Sampling**
- Motion-based frame selection to avoid redundant processing
- Configurable sampling rates based on performance mode
- Automatic duplicate frame detection and skipping

### **Memory Optimization**
- Automatic memory cleanup every 50 frames
- MPS cache clearing for Apple Silicon devices
- Efficient batch processing for large videos

### **Device Support**
- **Apple Silicon**: Automatic MPS acceleration
- **NVIDIA GPUs**: CUDA support
- **CPU Fallback**: Works on any system

## 🎯 Use Cases
- **Content Analysis** – Automatic video content categorization and tagging
- **Accessibility** – Generate descriptions for visually impaired users
- **Video Search** – Create searchable metadata from video content
- **Security Analysis** – Automated surveillance video analysis
- **Media Production** – Content review and scene breakdown

## 📜 License
This project is licensed under the **MIT License**.

## 👨‍💻 Development
The system follows a modular architecture with clear separation of concerns:
- Object detection pipeline using YOLOv8
- Scene captioning with BLIP-2 transformer models
- Efficient video processing with OpenCV
- Smart memory management and device optimization