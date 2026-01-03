# Project Structure

## Root Directory Layout
```
.
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── *.mp4                  # Sample video files for testing
├── .git/                  # Git version control
└── .kiro/                 # Kiro AI assistant configuration
    └── steering/          # AI guidance documents
```

## Code Organization

### main.py
The entire application is contained in a single Python file with the following structure:

- **Process Functions**: Each multiprocessing component has its own function
  - `streamer()` - Video frame capture and queuing
  - `detector()` - Motion detection and bounding box extraction
  - `presenter()` - Video display with blur effects
  - `analytics_processor()` - Motion density analysis and logging

- **Utility Functions**:
  - `apply_blur()` - Applies different blur algorithms to detected regions
  - `main()` - Orchestrates multiprocessing setup and execution

## Data Flow Architecture
1. **Streamer Process** → `frame_queue` → **Detector Process**
2. **Detector Process** → `detection_queue` → **Presenter Process**
3. **Detector Process** → `analytics_queue` → **Analytics Process**

## File Conventions
- Video files are stored in the root directory
- Configuration is done via hardcoded variables in `main.py`
- Logging uses Python's built-in logging module with INFO level
- No external configuration files - all settings are code-based

## Development Guidelines
- Single-file architecture keeps the codebase simple and self-contained
- Each process function should handle its own error cases and cleanup
- Queue-based communication ensures process isolation
- All video processing parameters are configurable through function arguments