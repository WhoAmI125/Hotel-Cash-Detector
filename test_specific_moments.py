"""
FAST TEST - Process only specific time moments where events occur
This tests the violence split logic in ~2-5 minutes instead of 30+ minutes
"""

import sys
from pathlib import Path
from app import process_video
import json
import uuid

# Configuration
VIDEO_FILE = "uploads/ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi"

# Specific time ranges to test (in seconds)
# Format: (start_seconds, end_seconds, description)
TEST_MOMENTS = [
    (792, 797, "Cash exchange at 794s"),
    (1033, 1042, "Violence scene at 1035-1040s"),
    (1041, 1045, "Brief event at 1043s"),
    (2238, 2243, "Event at 2239-2241s"),
    (2278, 2282, "Brief event at 2280s"),
    (2809, 2813, "Event at 2811s"),
    (2826, 2831, "Event at 2828-2829s"),
    (2839, 2844, "Event at 2841-2842s"),
    (2850, 2856, "Event at 2853s"),
    (3205, 3209, "Event at 3207s"),
    (3248, 3258, "Violence at 3250-3256s"),
]

def create_test_video_clips():
    """Create a shorter test video with only the important moments"""
    import cv2
    
    print("=" * 70)
    print("🚀 FAST TEST - Specific Moments Only")
    print("=" * 70)
    print(f"📹 Source video: {VIDEO_FILE}")
    print(f"📊 Test moments: {len(TEST_MOMENTS)} time ranges")
    print()
    
    video_path = Path(VIDEO_FILE)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None, None
    
    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {total_frames} frames @ {fps} fps ({width}x{height})")
    print()
    
    # Create output path
    output_path = Path("uploads/test_moments_only.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_extracted = 0
    frame_mapping = []  # Maps new frame number to original time
    
    print("⚙️  Extracting specific moments...")
    for i, (start_sec, end_sec, desc) in enumerate(TEST_MOMENTS, 1):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        duration = end_sec - start_sec
        
        print(f"  {i:2d}. {desc:30s} | {start_sec:7.1f}s - {end_sec:7.1f}s ({duration:4.1f}s)")
        
        # Jump to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_mapping.append({
                'new_frame': total_extracted,
                'original_frame': frame_num,
                'original_time': frame_num / fps,
                'moment': i
            })
            total_extracted += 1
    
    cap.release()
    out.release()
    
    print()
    print(f"✅ Created test video: {output_path}")
    print(f"📊 Extracted {total_extracted} frames from {len(TEST_MOMENTS)} moments")
    print(f"⏱️  New video duration: {total_extracted/fps:.1f}s (vs {total_frames/fps:.0f}s original)")
    print()
    
    return output_path, frame_mapping

def main():
    # Create test video with only specific moments
    test_video_path, frame_mapping = create_test_video_clips()
    if not test_video_path:
        return 1
    
    # Load config
    config_path = Path("config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("📋 Current Settings:")
    print(f"   MERGE_CLIPS_WITHIN_SECONDS: {config.get('MERGE_CLIPS_WITHIN_SECONDS', 30)}s")
    print(f"   MAX_CASH_TRANSACTION_SECONDS: {config.get('MAX_CASH_TRANSACTION_SECONDS', 10)}s")
    print(f"   VIOLENCE_VELOCITY_THRESHOLD: {config.get('VIOLENCE_VELOCITY_THRESHOLD', 25)} px/f")
    print(f"   Violence split gap: 3.0s (in app.py)")
    print()
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    print(f"🔑 Job ID: {job_id}")
    print()
    
    # Process test video
    print("⚙️  Processing test video with specific moments...")
    print("    (This should take 2-5 minutes instead of 30+ minutes)")
    print()
    
    try:
        # Call the process_video function
        process_video(job_id, str(test_video_path), test_video_path.name, options=None)
        
        # Read results from output folder
        output_dir = Path(f"outputs/{job_id}")
        if not output_dir.exists():
            print(f"⚠️  Output directory not found: {output_dir}")
            return 1
        
        # Count clips
        violence_clips = list(output_dir.glob("*_VIOLENCE_*.mp4"))
        cash_clips = list(output_dir.glob("*_CASH_*.mp4"))
        fire_clips = list(output_dir.glob("*_FIRE_*.mp4"))
        
        total_clips = len(violence_clips) + len(cash_clips) + len(fire_clips)
        
        print()
        print("=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Total clips generated: {total_clips}")
        print()
        
        print("📈 Clips by type:")
        print(f"   🚨 VIOLENCE: {len(violence_clips)} clip(s)")
        print(f"   💵 CASH_EXCHANGE: {len(cash_clips)} clip(s)")
        print(f"   🔥 FIRE: {len(fire_clips)} clip(s)")
        print()
        
        # Show all clip details
        all_clips = []
        for clip in violence_clips:
            all_clips.append(('VIOLENCE', clip))
        for clip in cash_clips:
            all_clips.append(('CASH', clip))
        for clip in fire_clips:
            all_clips.append(('FIRE', clip))
        
        # Sort by timestamp
        all_clips.sort(key=lambda x: x[1].name)
        
        if all_clips:
            print("📋 All Clips (in order):")
            for i, (clip_type, clip) in enumerate(all_clips, 1):
                # Parse timestamp from filename
                parts = clip.stem.split('_')
                if len(parts) >= 4:
                    timestamp = parts[-1].replace('s', '')
                    icon = "🚨" if clip_type == "VIOLENCE" else "💵" if clip_type == "CASH" else "🔥"
                    print(f"  {i:2d}. {icon} {clip_type:10s} @ {timestamp:>6s}s - {clip.name}")
                else:
                    icon = "🚨" if clip_type == "VIOLENCE" else "💵" if clip_type == "CASH" else "🔥"
                    print(f"  {i:2d}. {icon} {clip_type:10s} - {clip.name}")
        
        print()
        print("=" * 70)
        print("🎯 TEST RESULTS:")
        print("=" * 70)
        print(f"Expected: {len(TEST_MOMENTS)} moments tested")
        print(f"Actual:   {total_clips} clips generated")
        print()
        
        if len(violence_clips) >= 5:
            print("🎉 SUCCESS! Violence clips are NOT being merged!")
            print(f"   Got {len(violence_clips)} separate violence clips")
        elif len(violence_clips) == 1:
            print("❌ FAILURE! Violence clips still merging into 1 clip")
            print("   → The 3-second gap logic is not working")
        elif len(violence_clips) > 1:
            print(f"⚠️  PARTIAL SUCCESS: Got {len(violence_clips)} violence clips")
            print(f"   (Expected 5-8 separate violence incidents)")
        else:
            print("❓ No violence clips detected")
        
        print()
        print("=" * 70)
        print(f"📁 Output location: {output_dir}")
        print(f"📹 Test video kept at: {test_video_path}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR during processing:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
