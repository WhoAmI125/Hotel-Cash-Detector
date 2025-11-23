"""
FAST Test - Process only frames 48000-49500 (contains the main violence scene)
This tests the split logic without waiting 30+ minutes
"""

import sys
from pathlib import Path
import json
import cv2
from main import SimpleHandTouchConfig, SimpleHandTouchDetector
from multi_detector import MultiEventDetector

VIDEO_FILE = "uploads/ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi"
START_FRAME = 48000  # Start at frame 48000 (around violence scene)
END_FRAME = 49500    # Process 1500 frames (100 seconds @ 15fps)

def main():
    print("=" * 70)
    print("🚀 FAST TEST - Violence Split Logic (Partial Video)")
    print("=" * 70)
    print(f"📹 Video: {VIDEO_FILE}")
    print(f"📊 Processing frames: {START_FRAME} to {END_FRAME} ({END_FRAME-START_FRAME} frames)")
    print()
    
    video_path = Path(VIDEO_FILE)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return 1
    
    # Load config
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("📋 Settings:")
    print(f"   MERGE_CLIPS_WITHIN_SECONDS: {config.get('MERGE_CLIPS_WITHIN_SECONDS', 30)}s")
    print(f"   Violence split gap: 3.0s (hardcoded in app.py)")
    print()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {total_frames} frames @ {fps} fps")
    print()
    
    # Initialize detectors
    print("⚙️  Initializing detectors...")
    cash_detector = SimpleHandTouchDetector(SimpleHandTouchConfig())
    multi_detector = MultiEventDetector(config)
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    
    # Track events
    current_events = {}
    all_events = []
    violence_detections = 0
    
    print(f"🎬 Processing frames {START_FRAME} to {END_FRAME}...")
    print()
    
    for frame_num in range(START_FRAME, END_FRAME):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show progress
        if (frame_num - START_FRAME) % 150 == 0:
            progress = ((frame_num - START_FRAME) / (END_FRAME - START_FRAME)) * 100
            print(f"  ⚡ Progress: {frame_num}/{END_FRAME} ({progress:.0f}%)")
        
        # Detect
        detections_dict = multi_detector.detect_all(frame, fps, frame_num)
        
        # Process violence detections
        violence_detections_list = detections_dict.get('VIOLENCE', [])
        if violence_detections_list and len(violence_detections_list) > 0:
            violence_detections += 1
            
            if 'VIOLENCE' not in current_events:
                # New violence event
                current_events['VIOLENCE'] = {
                    'type': 'VIOLENCE',
                    'start_frame': frame_num,
                    'end_frame': frame_num,
                    'start_time': frame_num / fps,
                    'end_time': frame_num / fps,
                    'count': 1
                }
            else:
                # Check gap
                last_frame = current_events['VIOLENCE']['end_frame']
                gap_seconds = (frame_num - last_frame) / fps
                
                if gap_seconds > 3.0:
                    # GAP > 3s: Save current event and start new one
                    print(f"  🔄 Violence gap detected: {gap_seconds:.1f}s > 3.0s - Creating NEW event")
                    all_events.append(current_events['VIOLENCE'])
                    current_events['VIOLENCE'] = {
                        'type': 'VIOLENCE',
                        'start_frame': frame_num,
                        'end_frame': frame_num,
                        'start_time': frame_num / fps,
                        'end_time': frame_num / fps,
                        'count': 1
                    }
                else:
                    # Continue existing event
                    current_events['VIOLENCE']['end_frame'] = frame_num
                    current_events['VIOLENCE']['end_time'] = frame_num / fps
                    current_events['VIOLENCE']['count'] += 1
    
    # Add remaining events
    for event in current_events.values():
        all_events.append(event)
    
    cap.release()
    
    print()
    print("=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"📊 Total violence detections: {violence_detections} frames")
    print(f"📊 Total violence events: {len(all_events)}")
    print()
    
    if all_events:
        print("📋 Violence Events:")
        for i, event in enumerate(all_events, 1):
            duration = event['end_time'] - event['start_time']
            frame_count = event.get('count', 0)
            print(f"{i:2d}. 🚨 {event['start_time']:7.1f}s - {event['end_time']:7.1f}s ({duration:5.1f}s, {frame_count} frames)")
    
    print()
    print("=" * 70)
    print("🎯 TEST RESULTS:")
    print("=" * 70)
    
    if len(all_events) >= 3:
        print("✅ SUCCESS! Violence events are splitting correctly!")
        print(f"   Got {len(all_events)} separate events (expected 3-5 in this range)")
    elif len(all_events) == 1:
        print("❌ FAILURE! Still merging into 1 event")
        print("   → The 3-second gap logic is not working")
    else:
        print(f"⚠️  PARTIAL: Got {len(all_events)} events (expected 3-5)")
    
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
