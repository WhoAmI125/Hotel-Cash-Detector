# Hotel Cash Transaction Detector - Web App

A Flask web application that automatically detects cash transactions in hotel surveillance videos and extracts video clips around those transactions.

---

## 📚 Documentation Quick Links

- **[🚀 Quick Start Guide](QUICKSTART.md)** - Get up and running in 3 steps
- **[📋 Project Summary](PROJECT_SUMMARY.md)** - Complete feature overview
- **[🏗️ Architecture](ARCHITECTURE.md)** - Technical design and implementation details
- **[📖 This File](README.md)** - Comprehensive documentation (you are here)

---

## Features

- 🎥 Upload up to 5 videos simultaneously
- 🤖 Automatic hand-to-hand transaction detection using YOLO pose estimation
- ✂️ Extracts video clips with configurable padding (2 seconds before/after)
- 📊 Real-time progress tracking for each video
- 💾 Download individual clips or all clips at once
- 🎨 Modern UI with Tailwind CSS (matching CCTV admin interface theme)

## How It Works

1. **Upload Videos**: Select up to 5 surveillance videos (MP4, AVI, MOV, MKV)
2. **Detection**: The system uses YOLOv8 pose estimation to detect when people's hands touch (indicating cash exchange)
3. **Clip Extraction**: For each detected transaction, the system extracts a video clip with 2 seconds before and 2 seconds after the event
4. **Download**: View and download all extracted clips

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model Files Are Present

Make sure the YOLO model files are in the `models/` folder:
- `models/yolov8s-pose.pt` (required)

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Click "Select Videos" or drag and drop up to 5 video files
3. Click "Start Analysis"
4. Monitor the progress of each video in real-time
5. Once complete, download individual clips or all clips at once

### Folder Structure

```
Hotel-Cash-Detector/
├── app.py                  # Flask application
├── main.py                 # Original batch processing script
├── templates/              # HTML templates
│   ├── index.html         # Upload page
│   └── results.html       # Results page
├── models/                 # YOLO model files
│   └── yolov8s-pose.pt
├── uploads/               # Uploaded videos (auto-created)
├── outputs/               # Extracted clips (auto-created)
└── requirements.txt       # Python dependencies
```

## Configuration

You can adjust the clip extraction settings in `app.py`:

```python
class TransactionClipExtractor:
    def __init__(self, config=None):
        # Clip extraction settings
        self.SECONDS_BEFORE = 2  # seconds before transaction
        self.SECONDS_AFTER = 2   # seconds after transaction
        
        # Detection settings
        self.config = {
            'HAND_TOUCH_DISTANCE': 80,  # pixels
            'POSE_CONFIDENCE': 0.5,
            'MIN_TRANSACTION_FRAMES': 3
        }
```

## Camera Configuration

For camera-specific settings, you can modify the config in `app.py` or extend it to load from JSON files (like `main.py` does for batch processing).

## API Endpoints

- `GET /` - Main upload page
- `POST /upload` - Upload videos for processing
- `GET /status/<job_id>` - Get processing status
- `GET /download/<job_id>/<filename>` - Download extracted clip
- `GET /results/<job_id>` - View results page

## Technical Details

### Detection Algorithm

1. **Pose Detection**: Uses YOLOv8-pose to detect people and their hand positions (wrist keypoints)
2. **Distance Calculation**: Measures distance between hands of different people
3. **Transaction Detection**: When hands are within threshold distance (80px default) for minimum frames (3 frames)
4. **Temporal Filtering**: Merges close transactions and filters false positives

### Clip Extraction

- Adds 2-second buffer before the first detected frame
- Adds 2-second buffer after the last detected frame
- Maintains original video quality and framerate
- Names clips with timestamp for easy identification

## Future Enhancements (Not Yet Implemented)

- Real-time camera stream processing
- WebSocket for live progress updates
- Multi-camera synchronization
- Advanced analytics dashboard
- User authentication
- Database storage for results
- Cloud deployment support

## Troubleshooting

### Model Not Found Error
Make sure `models/yolov8s-pose.pt` exists. If not, the first run will download it automatically.

### Out of Memory
Processing multiple large videos simultaneously may require significant RAM. Consider:
- Processing fewer videos at once
- Reducing video resolution before upload
- Using a more powerful machine

### Slow Processing
Video processing is CPU/GPU intensive. To improve speed:
- Ensure CUDA is available for GPU acceleration
- Use a smaller YOLO model (e.g., yolov8n-pose.pt)
- Process videos sequentially instead of in parallel

## License

This project uses the YOLO model from Ultralytics, which is licensed under AGPL-3.0.

## Credits

- YOLOv8 by Ultralytics
- Tailwind CSS for UI styling
- Font Awesome for icons
