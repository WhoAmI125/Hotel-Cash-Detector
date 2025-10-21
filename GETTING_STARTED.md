# Getting Started with Cash Transaction Detector

## 🎯 What This Application Does

This web application helps you:
1. **Upload** surveillance videos from your hotel
2. **Detect** cash transactions automatically (hand-to-hand exchanges)
3. **Extract** short video clips around each transaction
4. **Download** clips for review, evidence, or training

---

## 🖼️ Visual Guide

### Step 1: Upload Page
```
┌─────────────────────────────────────────────────┐
│  Hotel Cash Transaction Detector                │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │  📤  Drag and drop videos here          │   │
│  │                                          │   │
│  │  or click to browse                      │   │
│  │                                          │   │
│  │  [Select Videos Button]                  │   │
│  │                                          │   │
│  │  Max 5 videos • MP4, AVI, MOV, MKV      │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  Selected Videos:                                │
│  ☑ video1.mp4 (45.2 MB)  [x]                   │
│  ☑ video2.mp4 (32.8 MB)  [x]                   │
│                                                  │
│  [Start Analysis Button]                         │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Step 2: Processing View
```
┌─────────────────────────────────────────────────┐
│  ⚙️  Processing Videos                          │
├─────────────────────────────────────────────────┤
│                                                  │
│  📹 video1.mp4                                  │
│  Status: Detecting transactions...               │
│  [████████████░░░░░░░] 65%                      │
│                                                  │
│  📹 video2.mp4                                  │
│  Status: Extracting clips...                     │
│  [███████████████░░░] 85%                       │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Step 3: Results View
```
┌─────────────────────────────────────────────────┐
│  ✅ Analysis Complete                           │
├─────────────────────────────────────────────────┤
│                                                  │
│  video1.mp4                                      │
│  Found 3 transaction(s)                          │
│                                                  │
│  ┌─────────────────────────────────────┐       │
│  │ 💰 Transaction 1                     │       │
│  │ Time: 15.2s - 18.5s (3.3s)          │       │
│  │ [Download Clip]                      │       │
│  └─────────────────────────────────────┘       │
│                                                  │
│  ┌─────────────────────────────────────┐       │
│  │ 💰 Transaction 2                     │       │
│  │ Time: 42.1s - 45.8s (3.7s)          │       │
│  │ [Download Clip]                      │       │
│  └─────────────────────────────────────┘       │
│                                                  │
│  [Download All Clips (3)]                        │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start Commands

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check installation (optional)
python check_installation.py

# 3. Start application
python app.py
```

### Every Time You Use It
```bash
# Just run the app
python app.py

# Or use the launcher
# Windows: run_app.bat
# Mac/Linux: ./run_app.sh
```

### Open in Browser
```
http://localhost:5000
```

---

## 🎛️ Configuration Guide

### Default Settings (Good for Most Cases)
```json
{
  "HAND_TOUCH_DISTANCE": 80,
  "MIN_TRANSACTION_FRAMES": 3,
  "SECONDS_BEFORE_TRANSACTION": 2,
  "SECONDS_AFTER_TRANSACTION": 2
}
```

### For Strict Detection (Fewer False Positives)
```json
{
  "HAND_TOUCH_DISTANCE": 60,      // Hands must be closer
  "MIN_TRANSACTION_FRAMES": 5     // Must last longer
}
```

### For Lenient Detection (Catch More)
```json
{
  "HAND_TOUCH_DISTANCE": 100,     // Hands can be farther
  "MIN_TRANSACTION_FRAMES": 2     // Can be briefer
}
```

### For Poor Quality Videos
```json
{
  "POSE_CONFIDENCE": 0.3,          // Lower confidence threshold
  "HAND_TOUCH_DISTANCE": 120       // More lenient distance
}
```

---

## 📖 Documentation Roadmap

**New to the project?** Follow this reading order:

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** (This file) - Visual guide
2. **[QUICKSTART.md](QUICKSTART.md)** - Installation and first run
3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Feature overview
4. **[README.md](README.md)** - Detailed documentation
5. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep dive

---

## 🎬 Example Workflow

### Scenario: Review Today's Transactions

1. **Collect Videos**
   - Export from your CCTV system
   - Get today's footage (e.g., cam1_morning.mp4, cam1_afternoon.mp4)

2. **Upload to App**
   - Open http://localhost:5000
   - Drag and drop both videos
   - Click "Start Analysis"

3. **Wait for Processing**
   - Watch progress bars
   - 5-minute video ≈ 2-5 minutes processing

4. **Review Results**
   - See all detected transactions with timestamps
   - Download suspicious clips
   - Save for review or evidence

5. **Take Action**
   - Review clips with your team
   - Follow up on flagged transactions
   - Archive clips as needed

---

## 🎯 Best Practices

### ✅ DO:
- Test with short sample videos first
- Adjust settings based on your camera setup
- Keep videos under 10 minutes for faster processing
- Use good lighting and clear camera angles
- Process during off-peak hours for large batches

### ❌ DON'T:
- Upload extremely long videos (>1 hour)
- Process too many videos simultaneously on limited hardware
- Expect perfect detection in crowded or poor quality footage
- Delete original videos (keep them as source)

---

## 🔧 Troubleshooting Quick Fixes

### App Won't Start
```bash
# Check if port is free
# Change port in app.py if needed:
app.run(debug=True, port=5001)
```

### Dependencies Missing
```bash
pip install -r requirements.txt
```

### Out of Memory
```bash
# Process fewer videos at once
# Or use smaller videos
# Or close other applications
```

### Slow Processing
```bash
# Use GPU if available
# Or use faster model: yolov8n-pose.pt
# Or reduce video resolution
```

### No Transactions Detected
```json
// Adjust config.json:
{
  "HAND_TOUCH_DISTANCE": 120,  // Increase
  "MIN_TRANSACTION_FRAMES": 2  // Decrease
}
```

---

## 💡 Tips for Success

### Camera Setup
```
         📹 Camera (overhead view)
           |
           |
           v
    [Cashier] <----> [Customer]
      (hands)        (hands)
```

**Ideal Setup:**
- Overhead or slight angle view
- 2-3 meters height
- Centered on transaction area
- Good, consistent lighting
- Minimum 720p resolution

### Video Quality Checklist
- [ ] Can you clearly see people's hands?
- [ ] Is lighting consistent (no heavy shadows)?
- [ ] Is the camera stable (not shaking)?
- [ ] Is the transaction area in frame?
- [ ] Is resolution at least 720p?

---

## 🎓 Understanding Detection

### How It Works (Simple)
```
1. AI detects people in each frame
2. AI finds their hands (wrist positions)
3. Measures distance between hands
4. If hands are close → Transaction!
5. Extracts video clip with context
```

### What Gets Detected
- ✅ Direct hand-to-hand exchanges
- ✅ Cash handovers
- ✅ Receipt exchanges
- ✅ Close hand contact

### What Might Be Missed
- ⚠️ Transactions with hands obscured
- ⚠️ Very brief touches
- ⚠️ Transactions outside camera view
- ⚠️ Crowded scenes with overlapping people

---

## 🎉 You're Ready!

You now have everything you need to start detecting cash transactions in your hotel videos!

### Next Steps:
1. ✅ Run the app: `python app.py`
2. ✅ Open browser: http://localhost:5000
3. ✅ Upload a test video
4. ✅ Review the results
5. ✅ Adjust settings as needed

### Need Help?
- Check [QUICKSTART.md](QUICKSTART.md) for setup issues
- Read [README.md](README.md) for detailed docs
- Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for features

---

**Happy Detecting! 🎬🔍**

