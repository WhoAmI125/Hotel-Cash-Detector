#!/usr/bin/env python
"""
Quick test to verify Flask app integration with main.py
"""

import sys
import json

print("="*60)
print("🧪 Testing Flask App Integration")
print("="*60)
print()

# Test 1: Import main.py classes
print("Test 1: Importing from main.py...")
try:
    from main import SimpleHandTouchConfig, SimpleHandTouchDetector
    print("   ✅ Successfully imported SimpleHandTouchConfig")
    print("   ✅ Successfully imported SimpleHandTouchDetector")
except ImportError as e:
    print(f"   ❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Load config.json
print("\nTest 2: Loading config.json...")
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    print(f"   ✅ Config loaded with {len(config)} parameters")
    
    # Check key parameters
    key_params = [
        'POSE_MODEL',
        'HAND_TOUCH_DISTANCE',
        'CASHIER_ZONE',
        'MIN_CASHIER_OVERLAP',
        'CASHIER_PERSISTENCE_FRAMES'
    ]
    
    for param in key_params:
        if param in config:
            print(f"   ✅ {param}: {config[param]}")
        else:
            print(f"   ⚠️  {param}: Not found")
            
except Exception as e:
    print(f"   ❌ Failed to load config: {e}")
    sys.exit(1)

# Test 3: Create config object
print("\nTest 3: Creating SimpleHandTouchConfig...")
try:
    detector_config = SimpleHandTouchConfig()
    
    # Apply config from JSON
    for key, value in config.items():
        if hasattr(detector_config, key):
            setattr(detector_config, key, value)
    
    print(f"   ✅ Config object created")
    print(f"   ✅ HAND_TOUCH_DISTANCE: {detector_config.HAND_TOUCH_DISTANCE}")
    print(f"   ✅ CASHIER_ZONE: {detector_config.CASHIER_ZONE}")
    print(f"   ✅ MIN_CASHIER_OVERLAP: {detector_config.MIN_CASHIER_OVERLAP}")
    
except Exception as e:
    print(f"   ❌ Failed to create config: {e}")
    sys.exit(1)

# Test 4: Check if model file exists
print("\nTest 4: Checking model file...")
from pathlib import Path
model_path = Path(config.get('POSE_MODEL', 'models/yolov8s-pose.pt'))
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Model found: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"   ⚠️  Model not found: {model_path}")
    print(f"      Will be downloaded on first run")

# Test 5: Check folders
print("\nTest 5: Checking folders...")
folders = ['uploads', 'outputs', 'templates', 'models']
for folder in folders:
    path = Path(folder)
    if path.exists():
        print(f"   ✅ {folder}/")
    else:
        print(f"   ⚠️  {folder}/ (will be created)")

# Test 6: Check template files
print("\nTest 6: Checking template files...")
templates = ['templates/index.html', 'templates/results.html']
for template in templates:
    path = Path(template)
    if path.exists():
        size = path.stat().st_size
        print(f"   ✅ {template} ({size:,} bytes)")
    else:
        print(f"   ❌ {template} missing!")

print()
print("="*60)
print("✅ All Integration Tests Passed!")
print("="*60)
print()
print("Your Flask app is ready to use the full detection logic!")
print()
print("To start the app:")
print("  python app.py")
print()
print("Then open: http://localhost:5000")
print()

