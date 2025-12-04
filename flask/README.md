# Hotel CCTV Monitoring System

🏨 Real-time detection system for hotel CCTV cameras that detects:
- 💵 **Cash Transactions** - Detects hand-to-hand exchanges in the cashier zone
- ⚠️ **Violence** - Detects aggressive poses, fighting, and rapid movements
- 🔥 **Fire/Smoke** - Detects fire-colored regions and smoke


## Default Login Credentials

| Username | Password | Role | Description |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full system access |
| `pm_seoul` | `pm123` | Project Manager | Manages 강남점, 홍대점, 명동점 |
| `pm_gyeonggi` | `pm123` | Project Manager | Manages 판교점, 일산점 |
| `pm_busan` | `pm123` | Project Manager | Manages 해운대점, 서면점 |

> ⚠️ **Note**: Change default passwords in production!

## Features

- 🌐 **Web Dashboard** - Beautiful real-time monitoring interface
- 📹 **Video Upload** - Process uploaded videos or connect to CCTV
- ⚡ **Real-time Alerts** - WebSocket-based instant notifications
- 📊 **Detection Reports** - JSON reports with timestamps and details
- 🔄 **Background Detection** - Continuous camera monitoring even when not viewing
<!-- Cashier Zone Setup - Hidden from UI but works in backend -->
<!-- - 🎯 **Cashier Zone Setup** - Visual tool to define the cashier area -->

## Installation

### 1. Clone and Setup

```bash
cd Hotel-Cash-Detector
pip install -r requirements.txt
```

### 2. Download YOLO Models

The models will be downloaded automatically on first run. You can also manually place them in the `models/` folder:
- `yolov8n.pt` - Object detection
- `yolov8n-pose.pt` - Pose estimation (for cash/violence detection)

## Usage

### Option 1: Web Dashboard (Django)

Start the Django web application:

```bash
cd django_app
python manage.py runserver
```

Then open http://localhost:8000 in your browser.

**Features:**
- Multi-branch hotel monitoring
- Real-time video streaming with detection overlays
- Toggle detection types (Cash/Violence/Fire)
- View detection alerts and statistics
- Background detection workers

### Option 2: Background Detection Service

Run cameras continuously in the background with automatic event saving:

```bash
# From project root
python background_workers.py

# Or as Django management command
cd django_app
python manage.py run_camera_workers

# Process specific cameras
python manage.py run_camera_workers --cameras CAM-SEO-01,CAM-SEO-02

# Process all cameras from a branch
python manage.py run_camera_workers --branch 1
```

**Background Worker Features:**
- Runs all cameras continuously without UI
- Automatically detects Cash, Violence, Fire events
- Saves events to database with timestamps
- Records 1-minute video clips on detection
- Saves thumbnails for quick preview
- Auto-reconnects if stream disconnects
- Reloads camera settings every 30 seconds

**API Endpoints for Background Workers:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workers/status/` | GET | Get status of all workers |
| `/api/workers/start-all/` | POST | Start all camera workers |
| `/api/workers/stop-all/` | POST | Stop all camera workers |
| `/api/workers/<camera_id>/start/` | POST | Start worker for specific camera |
| `/api/workers/<camera_id>/stop/` | POST | Stop worker for specific camera |

### Option 3: Flask Standalone

Start the Flask web application:

```bash
cd flask
python app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Upload videos or use existing ones from `input/` folder
- Real-time video streaming with detection overlays
- Toggle detection types (Cash/Violence/Fire)
- View detection alerts and statistics
- Export detection reports

### Option 4: Command Line Processing

Process videos without the web interface:

```bash
# Process all videos in input folder
python process_videos.py

# Process a specific video
python process_videos.py input/video.mp4

# Process with live preview
python process_videos.py --preview

# Disable specific detection types
python process_videos.py --no-fire --no-violence
```

<!--
### Option 3: Setup Cashier Zone (Hidden from UI - backend only)

Before detecting cash transactions, define the cashier zone:

```bash
python setup_cashier_zone.py
```

This opens the first video and lets you draw a rectangle around the cashier area.
-->

## How Detection Works

### 💵 Cash Transaction Detection

Since we cannot directly detect Korean currency (no trained model), we detect the **action** of cash exchange:

1. **Pose Estimation** - Detect people and their hand positions using YOLOv8-Pose
2. **Cashier Zone** - Identify if a person is in the designated cashier area
3. **Hand Proximity** - Detect when hands from two people come close together
4. **Temporal Filtering** - Confirm transaction after consistent detection over multiple frames

### ⚠️ Violence Detection

Detects violent behavior through:

1. **Aggressive Poses** - Raised arms, punching motions
2. **Rapid Motion** - Sudden fast movements
3. **Close Combat** - Two people very close with aggressive indicators
4. **Fall Detection** - Person suddenly on ground

### 🔥 Fire/Smoke Detection

Uses computer vision techniques:

1. **Color Analysis** - Detect fire colors (red, orange, yellow) in HSV space
2. **Flickering Detection** - Analyze temporal variation (fire flickers)
3. **Smoke Detection** - Gray/white moving regions with background subtraction

## Project Structure

```
Hotel-Cash-Detector/
├── app.py                  # Flask web application
├── config.py               # Configuration settings
├── process_videos.py       # Command-line video processor
├── setup_cashier_zone.py   # Cashier zone calibration tool
├── requirements.txt        # Python dependencies
├── detectors/
│   ├── __init__.py
│   ├── base_detector.py    # Base detector class
│   ├── cash_detector.py    # Cash transaction detector
│   ├── violence_detector.py # Violence detector
│   ├── fire_detector.py    # Fire/smoke detector
│   └── unified_detector.py # Combined detector interface
├── templates/
│   └── index.html          # Web dashboard template
├── static/
│   ├── css/
│   └── js/
├── models/                 # YOLO models
│   ├── yolov8n.pt
│   └── yolov8n-pose.pt
├── input/                  # Input videos
├── uploads/                # Uploaded videos
└── output/                 # Processed videos and reports
```

## Configuration

Edit `config.py` to adjust settings:

```python
class DetectionConfig:
    CONFIDENCE_THRESHOLD = 0.5      # Detection confidence
    HAND_TOUCH_DISTANCE = 100       # Pixels for hand proximity
    MIN_TRANSACTION_FRAMES = 3      # Frames to confirm transaction
    VIOLENCE_CONFIDENCE = 0.6       # Violence detection threshold
    FIRE_CONFIDENCE = 0.5           # Fire detection threshold
    FRAME_SKIP = 2                  # Process every Nth frame
```

## Camera Configuration

Each camera folder can have a `config.json`:

```json
{
    "CAMERA_NAME": "camera1",
    "CASHIER_ZONE": [100, 200, 400, 300],
    "HAND_TOUCH_DISTANCE": 80,
    "POSE_CONFIDENCE": 0.5
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/upload` | POST | Upload video file |
| `/video_feed` | GET | Streaming video feed |
| `/stop_processing` | GET | Stop video processing |
| `/set_cashier_zone` | POST | Update cashier zone |
| `/toggle_detection` | POST | Toggle detection types |
| `/get_summary` | GET | Get detection summary |
| `/api/alerts` | GET | Get recent alerts |

## Tips for Best Results

1. **Camera Position** - Position cameras with a clear view of the cashier area
2. **Lighting** - Ensure adequate lighting for accurate pose detection
3. **Cashier Zone** - Draw the zone to include the transaction area, not too large
4. **Frame Skip** - Increase `FRAME_SKIP` for better performance on slower machines

## Limitations

- **Currency Detection** - Cannot directly detect Korean Won bills (no trained model)
- **Occlusion** - Detection accuracy decreases when people are occluded
- **Lighting** - Poor lighting affects pose estimation accuracy
- **Camera Distance** - Very distant cameras may have reduced accuracy

## License

MIT License
