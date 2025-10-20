"""
CASHIER ZONE CALIBRATION TOOL
Draw a rectangle on your video to define the cashier zone
"""

import cv2
import json
from pathlib import Path

class ZoneSelector:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangle"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def get_zone_coordinates(self):
        """Convert start/end points to [x, y, width, height]"""
        if not self.start_point or not self.end_point:
            return None
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        return [x, y, width, height]
    
    def select_zone(self, video_path):
        """Open video and let user draw cashier zone"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read video frame")
            cap.release()
            return None
        
        self.current_frame = frame.copy()
        
        # Create window
        window_name = "Draw Cashier Zone - Press 's' to save, 'r' to reset, 'q' to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "=" * 70)
        print("CASHIER ZONE CALIBRATION")
        print("=" * 70)
        print("Instructions:")
        print("  1. Click and drag to draw a rectangle around the CASHIER area")
        print("  2. Press 's' to SAVE the zone")
        print("  3. Press 'r' to RESET and draw again")
        print("  4. Press 'q' to QUIT without saving")
        print("=" * 70)
        print()
        
        zone = None
        
        while True:
            # Create display frame
            display_frame = self.current_frame.copy()
            
            # Draw current rectangle
            if self.start_point and self.end_point:
                cv2.rectangle(display_frame, self.start_point, self.end_point, 
                            (0, 255, 255), 3)
                
                # Draw coordinates
                zone = self.get_zone_coordinates()
                if zone:
                    text = f"Zone: x={zone[0]}, y={zone[1]}, w={zone[2]}, h={zone[3]}"
                    cv2.putText(display_frame, text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw instructions
            cv2.putText(display_frame, "Draw rectangle | 's' = Save | 'r' = Reset | 'q' = Quit", 
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                zone = self.get_zone_coordinates()
                if zone and zone[2] > 0 and zone[3] > 0:
                    print(f"✅ Zone saved: {zone}")
                    break
                else:
                    print("⚠️  Please draw a valid rectangle first")
                    
            elif key == ord('r'):  # Reset
                self.start_point = None
                self.end_point = None
                print("🔄 Reset - draw new zone")
                
            elif key == ord('q'):  # Quit
                print("❌ Cancelled")
                zone = None
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return zone


def main():
    """Main calibration workflow"""
    print("\n" + "=" * 70)
    print("🎯 CASHIER ZONE CALIBRATION TOOL")
    print("=" * 70)
    print()
    
    # Find all camera folders
    input_dir = Path("input")
    camera_folders = sorted([d for d in input_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("camera")])
    
    if not camera_folders:
        print("❌ No camera folders found in input/")
        print("Please create camera folders first (camera1, camera2, etc.)")
        return
    
    # Select camera
    print("Available cameras:")
    for idx, cam in enumerate(camera_folders, 1):
        print(f"  {idx}. {cam.name}")
    print()
    
    while True:
        try:
            choice = int(input("Select camera number: "))
            if 1 <= choice <= len(camera_folders):
                camera_folder = camera_folders[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(camera_folders)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Find videos in camera folder
    videos = list(camera_folder.glob("*.mp4"))
    if not videos:
        print(f"❌ No videos found in {camera_folder}")
        return
    
    # Use first video for calibration
    video_path = videos[0]
    print(f"\nUsing video: {video_path.name}")
    print()
    
    # Select zone
    selector = ZoneSelector()
    zone = selector.select_zone(str(video_path))
    
    if zone is None:
        print("\n❌ Calibration cancelled")
        return
    
    # Load or create config
    config_file = camera_folder / "config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\n✅ Loaded existing config from {config_file}")
    else:
        config = {
            "CAMERA_NAME": camera_folder.name,
            "HAND_TOUCH_DISTANCE": 80,
            "POSE_CONFIDENCE": 0.5,
            "MIN_TRANSACTION_FRAMES": 3,
            "CALIBRATION_SCALE": 1.0,
            "CAMERA_ANGLE": 0,
            "DRAW_HANDS": True,
            "DRAW_CONNECTIONS": True,
            "DEBUG_MODE": True
        }
        print(f"\n✅ Created new config")
    
    # Update cashier zone
    config["CASHIER_ZONE"] = zone
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ CALIBRATION COMPLETE!")
    print("=" * 70)
    print(f"Camera: {camera_folder.name}")
    print(f"Cashier Zone: {zone}")
    print(f"  - X: {zone[0]}")
    print(f"  - Y: {zone[1]}")
    print(f"  - Width: {zone[2]}")
    print(f"  - Height: {zone[3]}")
    print()
    print(f"💾 Configuration saved to: {config_file}")
    print()
    print("Next steps:")
    print("  1. Run 'python backup.py' to process videos")
    print("  2. The cashier zone will be shown as a yellow box")
    print("  3. Anyone in this zone will be identified as cashier")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

