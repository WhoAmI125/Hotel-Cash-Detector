# System Architecture

## Overview

The Hotel Cash Transaction Detector is a Flask-based web application that processes surveillance videos to automatically detect and extract cash transaction events.

## System Flow

```
┌─────────────────┐
│  User Browser   │
│  (Tailwind UI)  │
└────────┬────────┘
         │ Upload Videos
         ▼
┌─────────────────────────────┐
│   Flask Web Server          │
│   (app.py)                  │
│                             │
│  ┌─────────────────────┐   │
│  │ Upload Handler      │   │
│  │ - Validate files    │   │
│  │ - Generate job IDs  │   │
│  │ - Save to disk      │   │
│  └──────────┬──────────┘   │
│             │               │
│             ▼               │
│  ┌─────────────────────┐   │
│  │ Background Thread   │   │
│  │ (per video)         │   │
│  └──────────┬──────────┘   │
└─────────────┼───────────────┘
              │
              ▼
┌──────────────────────────────┐
│ TransactionClipExtractor     │
│                              │
│  ┌───────────────────────┐  │
│  │ 1. Detect Transactions│  │
│  │    - Load video       │  │
│  │    - YOLO pose detect │  │
│  │    - Track hands      │  │
│  │    - Find touches     │  │
│  └───────────┬───────────┘  │
│              │              │
│              ▼              │
│  ┌───────────────────────┐  │
│  │ 2. Extract Clips      │  │
│  │    - Calculate times  │  │
│  │    - Add buffers      │  │
│  │    - Write MP4 files  │  │
│  └───────────┬───────────┘  │
└──────────────┼───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Output Files                │
│  outputs/<job_id>/           │
│    - transaction_1_15s.mp4   │
│    - transaction_2_42s.mp4   │
│    - ...                     │
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Status Updates              │
│  (polling every 1s)          │
│  - Progress percentage       │
│  - Current stage             │
│  - Results/clips list        │
└──────────────────────────────┘
```

## Components

### 1. Frontend (HTML + Tailwind CSS)

**Files:**
- `templates/index.html` - Main upload page
- `templates/results.html` - Results display page

**Features:**
- Drag & drop file upload
- Real-time progress monitoring
- Responsive design matching admin interface theme
- Individual/bulk clip downloads

### 2. Backend (Flask)

**File:** `app.py`

**Key Classes:**

#### `TransactionClipExtractor`
- Handles video processing and transaction detection
- Uses YOLOv8-pose for pose estimation
- Extracts clips with configurable time buffers

**Key Routes:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Main upload page |
| `/upload` | POST | Handle video uploads |
| `/status/<job_id>` | GET | Get processing status (JSON) |
| `/download/<job_id>/<filename>` | GET | Download extracted clip |
| `/results/<job_id>` | GET | View results page |

### 3. Detection Engine

**Algorithm:**

```python
For each frame in video:
    1. Detect people using YOLO pose model
       - Get wrist keypoints (left/right hands)
    
    2. Calculate hand-to-hand distances
       - For all pairs of people
       - Check all 4 hand combinations (L-L, L-R, R-L, R-R)
    
    3. Detect transaction
       - If distance < HAND_TOUCH_DISTANCE (80px)
       - If sustained for MIN_TRANSACTION_FRAMES (3)
    
    4. Record transaction timestamp
       - start_frame, end_frame
       - start_time, end_time (seconds)

After processing all frames:
    5. Merge close transactions
       - Combine transactions within 1 second
    
    6. Extract clips
       - start = transaction_start - SECONDS_BEFORE (2s)
       - end = transaction_end + SECONDS_AFTER (2s)
       - Write to MP4 file
```

### 4. Configuration System

**File:** `config.json`

**Parameters:**

```json
{
  "POSE_MODEL": "models/yolov8s-pose.pt",
  "HAND_TOUCH_DISTANCE": 80,
  "POSE_CONFIDENCE": 0.5,
  "MIN_TRANSACTION_FRAMES": 3,
  "SECONDS_BEFORE_TRANSACTION": 2,
  "SECONDS_AFTER_TRANSACTION": 2,
  "MAX_VIDEOS_PER_UPLOAD": 5,
  "MAX_FILE_SIZE_MB": 500
}
```

## Data Flow

### Upload Phase

```
User → Browser → Flask → Disk
                    ↓
               Generate Job ID
                    ↓
            Start Background Thread
```

### Processing Phase

```
Background Thread:
    Read Video → YOLO Detection → Track Hands → 
    Detect Touches → Record Timestamps → 
    Extract Clips → Save to Disk
         ↓
    Update Status Dict
         ↓
    Browser Polls Status → Display Progress
```

### Results Phase

```
Processing Complete:
    Status Dict Updated with Clip Info
         ↓
    Browser Displays Results
         ↓
    User Downloads Clips
```

## File Structure

```
Hotel-Cash-Detector/
│
├── app.py                      # Flask application
├── main.py                     # Batch processing script
├── config.json                 # Configuration file
│
├── templates/                  # HTML templates
│   ├── index.html             # Upload page
│   └── results.html           # Results page
│
├── models/                     # YOLO models
│   └── yolov8s-pose.pt        # Pose estimation model
│
├── uploads/                    # Uploaded videos (temp)
│   └── <job_id>_<filename>
│
├── outputs/                    # Extracted clips
│   └── <job_id>/
│       ├── transaction_1.mp4
│       ├── transaction_2.mp4
│       └── ...
│
└── static/                     # Static assets (future use)
```

## Concurrency Model

- **Main Thread**: Flask web server, handles HTTP requests
- **Background Threads**: One per uploaded video, handles processing
- **Shared State**: `processing_status` dictionary (thread-safe for read/write)

## Performance Considerations

### Processing Speed
- Depends on: Video length, resolution, CPU/GPU
- Typical: 5-10 FPS on CPU, 20-30 FPS on GPU
- Example: 5-minute video = 2-5 minutes processing time

### Memory Usage
- Video frames held in memory during processing
- YOLO model: ~50MB GPU memory or ~200MB RAM
- Each video thread: ~500MB-2GB depending on resolution

### Scalability
- Current: Single-server, multi-threaded
- Future: Queue-based (Celery/RQ), distributed workers

## Security Considerations

1. **File Upload Validation**
   - Extension whitelist
   - Size limits
   - Secure filename sanitization

2. **Data Storage**
   - Temporary files cleaned up (manual for now)
   - Job-specific folders prevent conflicts
   - No database = no SQL injection risk

3. **Future Improvements**
   - Authentication/authorization
   - Rate limiting
   - HTTPS/TLS
   - Input sanitization for user-provided config

## Extension Points

### 1. Real-time Processing
Add WebSocket support for live camera feeds:
```python
from flask_socketio import SocketIO
# Emit progress updates via socket
```

### 2. Database Integration
Store results persistently:
```python
from flask_sqlalchemy import SQLAlchemy
# Track jobs, clips, users
```

### 3. Multi-camera Sync
Correlate transactions across multiple cameras:
```python
# Timestamp matching
# Spatial tracking across views
```

### 4. Advanced Analytics
```python
# Transaction patterns
# Time-of-day analysis
# Anomaly detection
```

## Deployment Options

### Development
```bash
python app.py
# Runs on localhost:5000
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

### Cloud (AWS/Azure/GCP)
- Upload to S3/Blob Storage
- Lambda/Functions for processing
- API Gateway for REST endpoints

## Monitoring & Logging

Currently: Console output

Future:
- Structured logging (JSON)
- Error tracking (Sentry)
- Performance monitoring (New Relic)
- Usage analytics

---

**This architecture balances simplicity for immediate use with extensibility for future enhancements.**

