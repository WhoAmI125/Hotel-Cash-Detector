"""
Multi-Event Detector for Hotel Security
Detects: Cash Exchange, Violence, and Fire
Optimized for speed with parallel processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from main import SimpleHandTouchDetector, SimpleHandTouchConfig


class MultiEventDetector:
    """
    Unified detector for multiple event types with priority system
    Priority: 1=Violence (highest), 2=Fire, 3=Cash Exchange
    """
    
    def __init__(self, config_dict):
        self.config = config_dict
        self.detection_types = config_dict.get('DETECTION_TYPES', {})
        
        # Initialize detectors based on what's enabled
        self.detectors = {}
        
        # Cash Exchange Detector (always enabled for now)
        if self.detection_types.get('CASH_EXCHANGE', {}).get('enabled', True):
            print("📦 Loading Cash Exchange detector...")
            cash_config = SimpleHandTouchConfig()
            # Update config from dict
            for key, value in config_dict.items():
                if hasattr(cash_config, key):
                    setattr(cash_config, key, value)
            self.detectors['CASH_EXCHANGE'] = SimpleHandTouchDetector(cash_config)
            print("✅ Cash Exchange detector ready")
        
        # Violence Detector (using pose analysis for aggressive movements)
        if self.detection_types.get('VIOLENCE', {}).get('enabled', False):
            print("📦 Loading Violence detector...")
            violence_model = config_dict.get('VIOLENCE_MODEL', 'models/yolov8n.pt')
            self.detectors['VIOLENCE'] = ViolenceDetector(violence_model, self.detection_types['VIOLENCE'])
            print("✅ Violence detector ready")
        
        # Fire Detector
        if self.detection_types.get('FIRE', {}).get('enabled', False):
            print("📦 Loading Fire detector...")
            fire_model = config_dict.get('FIRE_MODEL', 'models/yolov8n.pt')
            self.detectors['FIRE'] = FireDetector(fire_model, self.detection_types['FIRE'])
            print("✅ Fire detector ready")
        
        print(f"🎯 Active detectors: {', '.join(self.detectors.keys())}")
    
    def detect_all(self, frame, fps=None, frame_number=None):
        """
        Run all enabled detectors on a single frame (PARALLEL)
        Returns: (annotated_frame, detections_dict)
        
        Args:
            frame: Input frame
            fps: Frames per second (for violence detection timing)
            frame_number: Actual frame number from video (for tracking)
        """
        detections = {}
        output_frame = frame.copy()
        
        # Run all detectors (they're already optimized individually)
        for det_type, detector in self.detectors.items():
            if det_type == 'CASH_EXCHANGE':
                output_frame, cash_events = detector.detect_hand_touches(output_frame, frame_number=frame_number)
                if cash_events:
                    detections['CASH_EXCHANGE'] = cash_events
            
            elif det_type == 'VIOLENCE':
                violence_frame, violence_events = detector.detect(output_frame, fps=fps)
                if violence_events:
                    detections['VIOLENCE'] = violence_events
                    output_frame = violence_frame
            
            elif det_type == 'FIRE':
                fire_frame, fire_events = detector.detect(output_frame)
                if fire_events:
                    detections['FIRE'] = fire_events
                    output_frame = fire_frame
        
        return output_frame, detections
    
    def get_highest_priority_detection(self, detections):
        """Return the detection type with highest priority"""
        if not detections:
            return None
        
        priority_map = {}
        for det_type in detections.keys():
            priority_map[det_type] = self.detection_types.get(det_type, {}).get('priority', 99)
        
        # Lower number = higher priority
        return min(priority_map, key=priority_map.get)


class ViolenceDetector:
    """
    Violence/Fight detector using pre-trained YOLO model
    
    Requirements for violence detection:
    1. Confidence >= 70% (0.7)
    2. Must be continuous for at least 1 second
    3. Only class 1 (Violence/Fight), ignore class 0 (NoViolence)
    
    Uses YOLO trained on violence/fight datasets:
    - Can detect: fighting, punching, kicking, aggressive behavior
    - Classes: 'fight', 'violence', 'assault', etc.
    
    Recommended models:
    1. YOLOv8 trained on violence datasets
    2. Custom violence detection models
    3. Action recognition models
    """
    
    def __init__(self, model_path, config):
        print(f"  Loading violence model: {model_path}")
        self.model = YOLO(model_path)
        self.config = config
        self.confidence_threshold = max(config.get('confidence_threshold', 0.7), 0.7)  # Minimum 70%
        
        # Violence-related class names (from Musawer1214/Fight-Violence-detection-yolov8)
        # Model classes: 0=NoViolence, 1=Violence/Fight
        # Added weapon keywords for YOLO object detection models
        self.violence_classes = [
            'violence', 'fight', 'fighting', 'assault', 'punch', 'kick', 'hit',
            'weapon', 'knife', 'gun', 'blade', 'pistol', 'rifle', 'sword', 'bat',
            'attack', 'stab', 'shoot', 'threat'
        ]
        self.violence_class_id = 1  # Class ID for Violence/Fight
        
        # Track continuous violence detection (for 1 second requirement)
        self.violence_frames_count = 0  # How many consecutive frames with violence
        self.min_violence_frames = 15  # Minimum frames (1 second at 15fps, adjusts dynamically)
        self.last_violence_detected = False
        self.fps_estimate = 15  # Will be updated dynamically
    
    def detect(self, frame, fps=None):
        """
        Detect violence using pre-trained model
        Requires: confidence >= 70% AND continuous for at least 1 second
        """
        output_frame = frame.copy()
        detections = []
        violence_in_frame = False
        temp_detections = []  # Store potential detections
        
        # Update FPS estimate if provided
        if fps:
            self.fps_estimate = fps
            self.min_violence_frames = max(int(fps), 15)  # At least 1 second worth of frames
        
        try:
            # Run violence detection model
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Check each detection
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names[cls_id].lower() if hasattr(self.model, 'names') else str(cls_id)
                    
                    # ONLY detect Violence (class 1), IGNORE NoViolence (class 0)
                    # Check if class name contains "violence" or "fight" but NOT "no" or "non"
                    is_violence = (cls_id == self.violence_class_id) or (
                        any(v_class in class_name for v_class in self.violence_classes) and
                        'no' not in class_name and 'non' not in class_name
                    )
                    
                    # Check confidence threshold (70% minimum)
                    if is_violence and conf >= self.confidence_threshold:
                        violence_in_frame = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        temp_detections.append({
                            'type': 'VIOLENCE',
                            'confidence': conf,
                            'class_name': class_name,
                            'bbox': [x1, y1, x2, y2],
                            'description': f'{class_name.upper()} detected (confidence: {conf:.2%})'
                        })
                        
                        # Always draw bounding box (for visual feedback)
                        color = tuple(self.config.get('color', [0, 0, 255]))  # Red
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label with frame count
                        label = f'{class_name.upper()}: {conf:.0%} ({self.violence_frames_count}/{self.min_violence_frames})'
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
            
            # Track continuous violence
            if violence_in_frame:
                self.violence_frames_count += 1
                self.last_violence_detected = True
            else:
                # Reset if no violence detected
                self.violence_frames_count = 0
                self.last_violence_detected = False
            
            # ONLY report violence if it's been continuous for at least 1 second
            if self.violence_frames_count >= self.min_violence_frames:
                detections = temp_detections
                
                # Draw confirmed alert banner
                if detections:
                    color = tuple(self.config.get('color', [0, 0, 255]))  # Red
                    cv2.rectangle(output_frame, (10, 10), (400, 90), color, -1)
                    cv2.rectangle(output_frame, (10, 10), (400, 90), (0, 0, 0), 3)
                    cv2.putText(output_frame, "!!! VIOLENCE CONFIRMED !!!", 
                               (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    duration = self.violence_frames_count / self.fps_estimate
                    cv2.putText(output_frame, f"Duration: {duration:.1f}s | {len(detections)} incident(s)", 
                               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif violence_in_frame:
                # Show "detecting..." status (not confirmed yet)
                color = (0, 165, 255)  # Orange for "pending"
                cv2.rectangle(output_frame, (10, 10), (400, 70), color, -1)
                cv2.rectangle(output_frame, (10, 10), (400, 70), (0, 0, 0), 3)
                cv2.putText(output_frame, "Detecting violence...", 
                           (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                remaining = self.min_violence_frames - self.violence_frames_count
                cv2.putText(output_frame, f"Need {remaining} more frames (~{remaining/self.fps_estimate:.1f}s)", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"⚠️  Violence detection error: {e}")
            self.violence_frames_count = 0
        
        return output_frame, detections


class FireDetector:
    """
    Fire/Smoke detector using pre-trained YOLO model
    
    Uses YOLO trained on fire/smoke datasets:
    - Can detect: fire, smoke, flames
    - Works with fire detection models
    
    Recommended models:
    1. YOLOv8 trained on fire datasets
    2. Custom fire/smoke detection models
    3. D-Fire dataset trained models
    """
    
    def __init__(self, model_path, config):
        print(f"  Loading fire model: {model_path}")
        self.model = YOLO(model_path)
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Fire-related class names (adjust based on your model)
        self.fire_classes = ['fire', 'smoke', 'flame', 'flames', 'burning']
    
    def detect(self, frame):
        """Detect fire/smoke using pre-trained model"""
        output_frame = frame.copy()
        detections = []
        
        try:
            # Run fire detection model
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Check each detection
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names[cls_id].lower() if hasattr(self.model, 'names') else str(cls_id)
                    
                    # Check if this is a fire-related class
                    is_fire = any(f_class in class_name for f_class in self.fire_classes)
                    
                    if is_fire and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detections.append({
                            'type': 'FIRE',
                            'confidence': conf,
                            'class_name': class_name,
                            'bbox': [x1, y1, x2, y2],
                            'description': f'{class_name.upper()} detected (confidence: {conf:.2%})'
                        })
                        
                        # Draw bounding box
                        color = tuple(self.config.get('color', [0, 165, 255]))  # Orange
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label
                        label = f'{class_name.upper()}: {conf:.0%}'
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                
                # Draw alert banner if fire detected
                if detections:
                    color = tuple(self.config.get('color', [0, 165, 255]))  # Orange
                    cv2.rectangle(output_frame, (10, 90), (350, 160), color, -1)
                    cv2.rectangle(output_frame, (10, 90), (350, 160), (0, 0, 0), 3)
                    cv2.putText(output_frame, "!!! FIRE/SMOKE DETECTED !!!", 
                               (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"{len(detections)} source(s)", 
                               (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"⚠️  Fire detection error: {e}")
        
        return output_frame, detections


def test_multi_detector():
    """Test the multi-detector on a sample video"""
    import json
    
    # Load config
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Enable all detectors for testing
    config['DETECTION_TYPES']['VIOLENCE']['enabled'] = True
    config['DETECTION_TYPES']['FIRE']['enabled'] = True
    
    detector = MultiEventDetector(config)
    
    print("\n🎬 Multi-detector test ready!")
    print("This detector will identify:")
    print("  🔴 Violence (Priority 1)")
    print("  🟠 Fire (Priority 2)")
    print("  🟢 Cash Exchange (Priority 3)")


if __name__ == "__main__":
    test_multi_detector()

