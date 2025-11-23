# Quick Test Instructions

## Created Files

### 1. `test_specific_moments.py` ⚡ RECOMMENDED - FAST!
Python script that extracts and tests ONLY the specific time moments where events occur.

**Usage:**
```bash
python test_specific_moments.py
```
OR just double-click:
```bash
run_fast_test.bat
```

**Features:**
- ⚡ **FAST:** 2-5 minutes instead of 30+ minutes
- Extracts only 11 specific time ranges (61 seconds total)
- Tests the violence split logic on actual event moments
- Shows expected vs actual results
- Counts clips by type (VIOLENCE, CASH, FIRE)

**Time ranges tested:**
- 794s (cash exchange)
- 1035-1040s (violence scene)
- 1043s, 2239-2241s, 2280s (brief events)
- 2811s, 2828-2829s, 2841-2842s, 2853s (events)
- 3207s, 3250-3256s (violence)

### 2. `test_quick_run.py` - Full Video Test
Python script that runs detection on your ENTIRE 60-minute video.

**Usage:**
```bash
python test_quick_run.py
```

**Features:**
- Tests the violence merge fix on full video
- Shows expected vs actual results
- ⚠️ **SLOW:** Takes 15-30 minutes
- Use only if you want to test the complete video

### 2. `run_test.bat`
Windows batch file that clears cache and runs the test.

**Usage:**
Just double-click `run_test.bat` or run:
```bash
run_test.bat
```

**What it does:**
1. Clears Python bytecode cache (*.pyc files and __pycache__ folders)
2. Runs `test_quick_run.py`
3. Shows results
4. Pauses so you can read the output

## Expected Results

### ✅ SUCCESS (Fix Working)
```
📈 Clips by type:
   🚨 VIOLENCE: 7-11 clip(s)
   💵 CASH_EXCHANGE: 3-5 clip(s)
   🔥 FIRE: 0 clip(s)

🎉 SUCCESS! Violence clips are NOT being merged!
```

### ❌ FAILURE (Still Merging)
```
📈 Clips by type:
   🚨 VIOLENCE: 1 clip(s)
   💵 CASH_EXCHANGE: 7 clip(s)
   🔥 FIRE: 0 clip(s)

❌ FAILURE! Violence clips still merging into 1 clip
   → Did you restart Python after clearing cache?
```

## What Was Fixed

### Problem
All violence events were being merged into a single massive clip because of line 439 in `app.py`:

```python
# OLD CODE (WRONG)
if (event.get('type') == current.get('type') and 
    event['start_time'] - current['end_time'] <= 1.0):  # ← Hardcoded 1 second
```

This merged all violence events within 1 second of each other.

### Solution
Updated `app.py` line 425-460 to NEVER merge violence events:

```python
# NEW CODE (FIXED)
# NEVER merge violence events - each attack is separate
if event.get('type') == 'VIOLENCE' or current.get('type') == 'VIOLENCE':
    merged.append(current)
    current = event.copy()
    continue

# For non-violence: merge if same type and within threshold
if (event.get('type') == current.get('type') and 
    event['start_time'] - current['end_time'] <= self.MERGE_THRESHOLD):
```

Now:
- **Violence clips:** NEVER merged (each attack = separate clip)
- **Cash clips:** Merged within 30s window (config setting)
- **Fire clips:** Merged within 30s window (config setting)

## Video Being Tested

**File:** `uploads/ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi`
- **Duration:** 60 minutes (54,000 frames @ 15 fps)
- **Expected violence scenes:** ~7-11 separate incidents
- **Expected cash transactions:** ~3-5 scenes

## Output Location

Results are saved to: `outputs/<job-id>/`

Each run generates a new job ID (UUID), so multiple test runs won't overwrite each other.

## Troubleshooting

### Still getting 1 merged clip?
1. **Clear cache:** Run `run_test.bat` (it clears cache automatically)
2. **Restart Python:** Close all Python processes and try again
3. **Check code:** Verify line 441 in `app.py` says `# NEVER merge violence`

### Script fails with error?
1. **Check video exists:** `uploads/ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi`
2. **Check config exists:** `config.json`
3. **Install dependencies:** `pip install -r requirements.txt`

### Processing takes too long?
- **Normal:** 60-minute video takes 10-30 minutes to process
- **Speed up:** Reduce video length or use smaller model
- **Background:** Script shows progress every 150 frames

## Manual Testing (Alternative)

If you prefer to test manually without the script:

1. **Clear cache:**
   ```bash
   for /d /r %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
   del /s /q *.pyc
   ```

2. **Run Flask app:**
   ```bash
   python app.py
   ```

3. **Upload video via web UI:**
   - Go to http://localhost:5000
   - Upload `ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi`
   - Check results page

4. **Check clips:**
   Look in `outputs/<job-id>/` for multiple `*_VIOLENCE_*.mp4` files
