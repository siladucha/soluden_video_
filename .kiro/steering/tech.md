# Technology Stack

## Core Technologies
- **Python 3.8+** - Primary programming language
- **OpenCV (cv2)** - Computer vision and video processing
- **NumPy** - Numerical computing and array operations
- **Multiprocessing** - Parallel processing for performance optimization

## Dependencies
```
numpy==2.2.3
opencv-python==4.11.0.86
```

## Build System
This is a Python script-based project with minimal build requirements.

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

## Common Commands

### Development
- **Install dependencies**: `pip install -r requirements.txt`
- **Run main application**: `python main.py`
- **Exit video display**: Press 'q' key during video playback

### Configuration
The application is configured by modifying parameters in `main.py`:
- `video_path` - Path to input video file
- `blur_type` - Blur algorithm ('gaussian', 'median', 'bilateral')
- `draw_boxes` - Boolean to show/hide bounding boxes

## Architecture Patterns
- **Multiprocessing Pipeline**: Four separate processes handle different aspects
- **Queue-based Communication**: Inter-process data exchange using multiprocessing.Queue()
- **Producer-Consumer Pattern**: Streamer produces frames, other processes consume
- **Modular Functions**: Each process has a dedicated function with clear responsibilities