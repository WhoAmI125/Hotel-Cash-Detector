# Configuration Guide

## Understanding Your Detection Settings

Your Flask app now uses the **exact same detection logic** as `main.py`, with full cashier zone support!

---

## 🎯 Cashier Zone Explanation

### What is CASHIER_ZONE?

The `CASHIER_ZONE` defines a rectangular area where cashiers are expected to be. Anyone in this zone is considered a cashier, and the system tracks transactions between cashiers and customers.

### Current Configuration

```json
"CASHIER_ZONE": [0, 300, 900, 430]
```

This means:
```
[x, y, width, height]
[0, 300, 900, 430]
 │   │    │    │
 │   │    │    └─ Height: 430 pixels
 │   │    └────── Width: 900 pixels
 │   └─────────── Y position: 300 pixels from top
 └─────────────── X position: 0 pixels from left
```

### Visual Representation

```
Screen Coordinates (pixels)
0,0 ─────────────────────────► X-axis
│
│        ┌─────────────────────────────┐
│        │                             │
│        │   (Video Frame Area)        │
│        │                             │
│  300px │   ┌──────────────────────┐  │ ← CASHIER_ZONE starts here
│        │   │   CASHIER ZONE       │  │
│        │   │   (900 x 430 pixels) │  │
│        │   │                      │  │
│  730px │   └──────────────────────┘  │ ← Zone ends at Y=730 (300+430)
│        │                             │
│        └─────────────────────────────┘
▼
Y-axis
```

### How to Adjust Your Zone

1. **Open your video** in a video player or editor
2. **Note the pixel coordinates** where your cashier stands:
   - Measure from top-left corner (0,0)
   - X = horizontal position
   - Y = vertical position
3. **Update config.json**:
   ```json
   "CASHIER_ZONE": [x, y, width, height]
   ```

#### Example Scenarios

**Cashier on Left Side:**
```json
"CASHIER_ZONE": [0, 200, 400, 600]
```

**Cashier on Right Side:**
```json
"CASHIER_ZONE": [800, 200, 400, 600]
```

**Cashier in Center:**
```json
"CASHIER_ZONE": [300, 200, 600, 600]
```

**Full Width (Bottom Half):**
```json
"CASHIER_ZONE": [0, 540, 1920, 540]
```
(For 1080p video: 1920x1080)

---

## 📋 All Configuration Parameters

### Detection Settings

#### HAND_TOUCH_DISTANCE
```json
"HAND_TOUCH_DISTANCE": 100
```
- **What**: Maximum distance (in pixels) between hands to count as a transaction
- **Default**: 80
- **Your Setting**: 100 (more lenient)
- **Lower (50-70)**: Stricter detection, hands must be very close
- **Higher (100-150)**: More lenient, detects transactions from farther away

#### POSE_CONFIDENCE
```json
"POSE_CONFIDENCE": 0.5
```
- **What**: Minimum confidence for detecting a person (0.0 to 1.0)
- **Default**: 0.5
- **Lower (0.3-0.4)**: Detect partially visible people
- **Higher (0.6-0.8)**: Only detect clearly visible people

#### MIN_TRANSACTION_FRAMES
```json
"MIN_TRANSACTION_FRAMES": 1
```
- **What**: Minimum number of consecutive frames for a valid transaction
- **Default**: 3
- **Your Setting**: 1 (detects very brief touches)
- **Lower (1-2)**: Catch brief interactions
- **Higher (5-10)**: Only sustained hand contact

### Cashier Detection

#### CASHIER_ZONE
```json
"CASHIER_ZONE": [0, 300, 900, 430]
```
- **What**: Rectangular area [x, y, width, height] where cashiers are located
- **null**: Auto-detect (person at bottom of frame = cashier)
- **[x,y,w,h]**: Manual zone definition

#### MIN_CASHIER_OVERLAP
```json
"MIN_CASHIER_OVERLAP": 0.7
```
- **What**: Minimum percentage of person's body that must be in cashier zone
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Your Setting**: 0.7 (70% of body must be in zone)
- **Lower (0.3)**: Person just needs to touch the zone
- **Higher (0.9)**: Almost entire person must be in zone

#### CASHIER_PERSISTENCE_FRAMES
```json
"CASHIER_PERSISTENCE_FRAMES": 20
```
- **What**: How many frames to keep cashier status after leaving zone
- **Your Setting**: 20 frames (~0.67 seconds at 30fps)
- **Purpose**: Prevents flickering when cashier briefly steps out

### Visualization (Not Used in Flask App)

These are from `main.py` and used when saving annotated videos:

```json
"DRAW_HANDS": true,
"DRAW_CONNECTIONS": true,
"DEBUG_MODE": true
```

### Clip Extraction

#### SECONDS_BEFORE_TRANSACTION
```json
"SECONDS_BEFORE_TRANSACTION": 2
```
- **What**: How many seconds before transaction to include in clip
- **Purpose**: Provides context (seeing people approach)

#### SECONDS_AFTER_TRANSACTION
```json
"SECONDS_AFTER_TRANSACTION": 2
```
- **What**: How many seconds after transaction to include in clip
- **Purpose**: Provides context (seeing people leave)

### App Limits

```json
"MAX_VIDEOS_PER_UPLOAD": 5,
"MAX_FILE_SIZE_MB": 500,
"ALLOWED_EXTENSIONS": ["mp4", "avi", "mov", "mkv"]
```

---

## 🔧 Common Adjustments

### Problem: Too Many False Positives

**Solution 1**: Increase strictness
```json
{
  "HAND_TOUCH_DISTANCE": 60,
  "MIN_TRANSACTION_FRAMES": 5,
  "MIN_CASHIER_OVERLAP": 0.8
}
```

**Solution 2**: Tighten cashier zone
```json
{
  "CASHIER_ZONE": [100, 300, 600, 400]  // Smaller zone
}
```

### Problem: Missing Transactions

**Solution 1**: Decrease strictness
```json
{
  "HAND_TOUCH_DISTANCE": 120,
  "MIN_TRANSACTION_FRAMES": 1,
  "MIN_CASHIER_OVERLAP": 0.5
}
```

**Solution 2**: Expand cashier zone
```json
{
  "CASHIER_ZONE": [0, 200, 1000, 600]  // Larger zone
}
```

### Problem: Cashier Not Detected

**Check 1**: Is cashier in the zone?
- Review your CASHIER_ZONE coordinates
- Make sure it covers where cashier actually stands

**Check 2**: Is overlap requirement too high?
```json
{
  "MIN_CASHIER_OVERLAP": 0.3  // Lower requirement
}
```

**Check 3**: Use auto-detection
```json
{
  "CASHIER_ZONE": null  // Auto-detect bottom person
}
```

---

## 🎬 Testing Your Configuration

### Step 1: Test with Short Clip
1. Extract a 30-second clip from your video
2. Upload to the app
3. Review results

### Step 2: Adjust Settings
1. Edit `config.json`
2. Restart the app: `python app.py`
3. Test again

### Step 3: Fine-Tune
- Iterate until you get desired accuracy
- Document your final settings

---

## 📊 Detection Logic Flow

```
For each frame:
  1. Detect all people using YOLO pose model
  2. Extract hand positions (wrist keypoints)
  3. Calculate bounding box for each person
  
  4. Identify Cashiers:
     - If CASHIER_ZONE defined:
       → Check if person's bbox overlaps zone
       → If overlap >= MIN_CASHIER_OVERLAP: CASHIER
     - If no CASHIER_ZONE:
       → Person with highest Y value (bottom): CASHIER
  
  5. For each CASHIER:
     For each CUSTOMER:
       - Calculate hand-to-hand distances (4 combinations)
       - If ANY distance <= HAND_TOUCH_DISTANCE:
         → TRANSACTION detected
  
  6. Track transactions over time:
     - Must last >= MIN_TRANSACTION_FRAMES
     - Merge close transactions (within 1 second)
  
  7. Extract clips:
     - Start: transaction_start - SECONDS_BEFORE
     - End: transaction_end + SECONDS_AFTER
```

---

## 💡 Pro Tips

### Tip 1: Use setup_cashier_zone.py
```bash
python setup_cashier_zone.py
```
This tool helps you visually define your cashier zone!

### Tip 2: Camera-Specific Configs
For multiple cameras, you can maintain separate configs:
```
config_camera1.json
config_camera2.json
config_camera3.json
```

Then in `app.py`, you could load based on camera selection.

### Tip 3: Test Different Lighting
Your zone might need adjustment for:
- Morning vs evening lighting
- Sunny vs cloudy days
- Different seasons

### Tip 4: Document Your Settings
Keep notes on what works:
```
Camera 1 (Front Desk):
- CASHIER_ZONE: [0, 300, 900, 430]
- HAND_TOUCH_DISTANCE: 100
- Works well during day shifts

Camera 2 (Side Desk):
- CASHIER_ZONE: [800, 200, 400, 600]
- HAND_TOUCH_DISTANCE: 80
- Better for night shifts
```

---

## 🎯 Quick Reference

| Setting | Purpose | Increase To... | Decrease To... |
|---------|---------|----------------|----------------|
| HAND_TOUCH_DISTANCE | Hand proximity threshold | Catch farther exchanges | More strict detection |
| MIN_TRANSACTION_FRAMES | Transaction duration | Ignore brief touches | Catch quick exchanges |
| POSE_CONFIDENCE | Person detection | Only clear people | Include partial visibility |
| MIN_CASHIER_OVERLAP | Zone coverage | Require more in zone | Allow zone edge |
| CASHIER_PERSISTENCE_FRAMES | Status retention | Longer persistence | Shorter persistence |

---

## 🚀 After Configuration

Once you're happy with your settings:

1. ✅ Save `config.json`
2. ✅ Restart app: `python app.py`
3. ✅ Upload test videos
4. ✅ Review extracted clips
5. ✅ Document successful configuration

Your transaction detection system is now tuned to your specific setup! 🎉

