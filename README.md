# Video Processing System

## 📌 Overview
This project is a **real-time video processing system** designed for **motion detection**, **blurring detected objects**, and **analyzing motion density**. It leverages **OpenCV** and **multiprocessing** to efficiently handle large-scale video streams.

## 🚀 Features
- **📹 Video Streaming** – Reads and processes video files frame-by-frame.
- **🔍 Motion Detection** – Uses OpenCV’s **BackgroundSubtractorMOG2** to detect moving objects.
- **🎭 Adaptive Blurring** – Applies **Gaussian, Median, or Bilateral blur** to detected areas.
- **📊 Motion Analysis** – Computes **motion density per second** and logs statistics.
- **⚡ Multiprocessing** – Efficiently handles each component in separate processes.
- **🛠️ Configurable Parameters** – Allows switching blur types and enabling/disabling bounding boxes.

## 🏗️ System Architecture
The system is built using **Python’s multiprocessing framework** with four independent processes:

1️⃣ **Streamer** – Reads video frames and pushes them into a queue.
2️⃣ **Detector** – Identifies moving objects and extracts bounding boxes.
3️⃣ **Presenter** – Displays the video with applied blur and optional bounding boxes.
4️⃣ **Analytics Processor** – Calculates and logs motion density over time.

Data is exchanged between processes using **multiprocessing.Queue()**.

## 🛠️ Installation & Setup
### **Prerequisites**
Ensure you have **Python 3.8+** installed.

### **Install Required Packages**
```sh
pip install -r requirements.txt
```

## 🚀 Usage
Run the script with a selected **video file** and **blurring type**:
```sh
python main.py
```

### **Configurable Parameters**
Modify the following parameters in `main.py`:
```python
video_path = "video.mp4"  # Change to your video file
blur_type = "gaussian"  # Options: 'gaussian', 'median', 'bilateral'
draw_boxes = False  # Set to True to show bounding boxes
main(video_path, blur_type, draw_boxes)
```

## 📊 Example Logs
The system logs motion density over time:
```
2025-02-26 12:45:00 - INFO - Motion density for 1.00 sec: 3 objects
2025-02-26 12:45:01 - INFO - Motion density for 1.00 sec: 5 objects
2025-02-26 12:45:02 - INFO - Average motion density: 4.2 objects per second
```

## 🔧 Future Enhancements
- ✅ **Live Video Streaming Support**
- ✅ **Dynamic Blurring Selection via Command Line Arguments**
- ✅ **Web Interface for Video Upload & Processing**

## 📜 License
This project is licensed under the **MIT License**.

## 👨‍💻 Author
Developed by **Max Bregert** for the **Senior Python Developer Assignment**.

