# 💰 Cash Transaction Detection System
**AI-Powered Hand-to-Hand Transaction Monitor**

---

## 🎯 **What It Does**

Automatically detects cash transactions between cashiers and customers by tracking hand-to-hand interactions using AI pose estimation.

### **Key Detection:**
- ✅ **Cashier identification** (zone-based or position-based)
- ✅ **Hand proximity detection** (when hands touch = transaction)
- ✅ **Multiple cashier support** (handles multiple staff simultaneously)
- ✅ **Temporal filtering** (confirms sustained transactions, reduces false positives)

---

## 🛠️ **Technology Stack**

- **AI Model**: YOLOv8 Pose Estimation (`yolov8s-pose.pt`)
- **Framework**: Ultralytics YOLO
- **Language**: Python 3.x
- **Libraries**: OpenCV, NumPy

---

## 📊 **How It Works**

### **Detection Pipeline:**

```
1. VIDEO INPUT
   ↓
2. POSE DETECTION (YOLOv8-Pose)
   → Detects people + hand keypoints
   ↓
3. CASHIER IDENTIFICATION
   → Zone-based: Anyone in yellow "CASHIER ZONE" box
   → Auto: Person at bottom of frame
   ↓
4. HAND PROXIMITY CHECK
   → Measures distance between cashier/customer hands
   → Threshold: configurable (default 80-100px)
   ↓
5. TEMPORAL VALIDATION
   → Confirms transaction if sustained 3+ frames
   ↓
6. OUTPUT: Annotated video + transaction log
```

---

## ⚙️ **Configuration System**

### **Camera-Based Setup:**
```
input/
├── camera1/
│   ├── config.json          ← Camera-specific settings
│   └── videos...
├── camera2/
│   ├── config.json
│   └── videos...
```

### **Key Config Parameters:**

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `HAND_TOUCH_DISTANCE` | Max hand distance (px) for transaction | 80-100 |
| `CASHIER_ZONE` | Defines cashier area `[x, y, w, h]` | Auto-detect |
| `MIN_TRANSACTION_FRAMES` | Frames to confirm transaction | 3 |
| `MIN_CASHIER_OVERLAP` | % of body in zone to be cashier | 0.3 (30%) |
| `CASHIER_PERSISTENCE_FRAMES` | Frames to keep cashier status | 20 (~0.7s) |

---

## 🚀 **Key Features**

### **1. Smart Cashier Detection**
- **Zone-based**: Define custom cashier area per camera
- **Bounding box overlap**: Detects cashiers even when leaning
- **Persistence**: Maintains cashier status during brief movements
- **Multi-cashier**: Supports multiple cashiers simultaneously

### **2. Robust Tracking**
- **Stable ID assignment**: Tracks same person across frames
- **IoU-based matching**: Uses bounding box overlap
- **Occlusion handling**: Continues tracking through brief occlusions

### **3. Transaction Validation**
- **Every cashier ↔ every customer**: Checks all possible pairs
- **4 hand combinations**: R-R, R-L, L-R, L-L
- **Temporal filtering**: Reduces false positives from accidental touches
- **Distance-based**: Configurable threshold per camera angle

### **4. Visual Output**
- **Color-coded hands**: 
  - 🟡 Gold = Cashier hands
  - 🔵 Blue/🔴 Red = Customer hands
- **Transaction indicators**: Green line between touching hands
- **Debug info**: Distance, status, frame count
- **Zone overlay**: Yellow box showing cashier area

---

## 📈 **Performance**

- **Processing Speed**: ~10-30 FPS (depending on resolution)
- **Accuracy**: High precision with proper calibration
- **Scalability**: Handles multiple cameras/locations

---

## 🔧 **Usage**

### **Setup:**
```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy

# 2. Organize videos
input/camera1/video1.mp4
input/camera1/config.json

# 3. Run detector
python main.py
```

### **Calibration Tool:**
```bash
# Interactive zone setup
python setup_cashier_zone.py
```

### **Output:**
```
output/
├── camera1/
│   └── hand_touch_video1.mp4  ← Annotated video
├── camera2/
│   └── hand_touch_video2.mp4
```

---

## 🎯 **Detection Logic**

### **Cashier Identification:**
1. Check if person's bounding box overlaps `CASHIER_ZONE`
2. Require minimum 30% overlap (configurable)
3. Maintain cashier status for 20 frames after leaving zone
4. Support multiple cashiers in zone

### **Transaction Detection:**
```python
For each (cashier, customer) pair:
    distance = min_hand_distance(cashier_hands, customer_hands)
    
    if distance <= HAND_TOUCH_DISTANCE:
        if sustained for MIN_TRANSACTION_FRAMES:
            ✅ CONFIRMED TRANSACTION
```

---

## 📝 **Configuration Example**

```json
{
  "CAMERA_NAME": "camera1",
  "HAND_TOUCH_DISTANCE": 100,
  "POSE_CONFIDENCE": 0.5,
  "MIN_TRANSACTION_FRAMES": 3,
  "CASHIER_ZONE": [0, 300, 900, 430],
  "CASHIER_PERSISTENCE_FRAMES": 20,
  "MIN_CASHIER_OVERLAP": 0.3,
  "DEBUG_MODE": true
}
```
