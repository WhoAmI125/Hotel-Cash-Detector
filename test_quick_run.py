"""
Quick Test Script - Run detection on specific video with updated merge logic
Processes: ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi
Expected: Multiple separate violence clips (not merged into 1)
"""

import sys
from pathlib import Path
from app import process_video
import json
import uuid

# Configuration
VIDEO_FILE = "uploads/ad53d6d1-c49e-43fd-a319-c0c750ce13d6_251111_1116.avi"
CONFIG_FILE = "config.json"

def main():
    print("=" * 70)
    print("🧪 QUICK TEST - Violence Merge Fix Verification")
    print("=" * 70)
    print(f"📹 Video: {VIDEO_FILE}")
    print()
    
    # Check if video exists
    video_path = Path(VIDEO_FILE)
    if not video_path.exists():
        print(f"❌ ERROR: Video not found at {video_path}")
        print(f"   Available videos in uploads/:")
        uploads = Path("uploads")
        if uploads.exists():
            for v in list(uploads.glob("*.avi"))[:5]:
                print(f"   - {v.name}")
        return 1
    
    # Load config
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"❌ ERROR: Config not found at {config_path}")
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("📋 Current Settings:")
    print(f"   MERGE_CLIPS_WITHIN_SECONDS: {config.get('MERGE_CLIPS_WITHIN_SECONDS', 30)}s")
    print(f"   MAX_CASH_TRANSACTION_SECONDS: {config.get('MAX_CASH_TRANSACTION_SECONDS', 10)}s")
    print(f"   VIOLENCE_VELOCITY_THRESHOLD: {config.get('VIOLENCE_VELOCITY_THRESHOLD', 25)} px/f")
    print()
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    print(f"🔑 Job ID: {job_id}")
    print()
    
    # Process video
    print("⚙️  Processing video (this may take several minutes)...")
    print()
    
    try:
        # Call the process_video function (runs in same thread for testing)
        process_video(job_id, str(video_path), video_path.name, options=None)
        
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
        
        # Show violence clip details
        if violence_clips:
            print("📋 Violence Clip Details:")
            for clip in sorted(violence_clips):
                # Parse timestamp from filename
                parts = clip.stem.split('_')
                if len(parts) >= 4:
                    timestamp = parts[-1].replace('s', '')
                    print(f"   🚨 {clip.name} (@ {timestamp}s)")
                else:
                    print(f"   🚨 {clip.name}")
        
        print()
        print("=" * 70)
        print("🎯 TEST EXPECTATIONS:")
        print("=" * 70)
        print("✅ PASS: If you see MULTIPLE violence clips (5-10+)")
        print("❌ FAIL: If you see only 1 violence clip (still merging)")
        print()
        print("Expected: ~7-11 separate clips (violence + cash)")
        print(f"Actual:   {total_clips} clips ({len(violence_clips)} violence)")
        print()
        
        if len(violence_clips) >= 5:
            print("🎉 SUCCESS! Violence clips are NOT being merged!")
        elif len(violence_clips) == 1:
            print("❌ FAILURE! Violence clips still merging into 1 clip")
            print("   → Did you restart Python after clearing cache?")
        else:
            print(f"⚠️  PARTIAL: Got {len(violence_clips)} clips but expected more")
        
        print("=" * 70)
        print(f"📁 Output location: {output_dir}")
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
