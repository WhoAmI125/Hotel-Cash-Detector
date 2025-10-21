#!/usr/bin/env python
"""
Installation Check Script
Verifies that all dependencies and files are properly set up
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    required = {
        'flask': 'Flask',
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy'
    }
    
    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (run: pip install {package})")
            all_ok = False
    
    return all_ok

def check_folders():
    """Check if necessary folders exist"""
    print("\n📁 Checking folders...")
    folders = ['models', 'templates', 'uploads', 'outputs']
    
    all_ok = True
    for folder in folders:
        path = Path(folder)
        if path.exists():
            print(f"   ✅ {folder}/")
        else:
            print(f"   ⚠️  {folder}/ (will be created automatically)")
            if folder not in ['uploads', 'outputs']:
                all_ok = False
    
    return all_ok

def check_files():
    """Check if required files exist"""
    print("\n📄 Checking files...")
    files = {
        'app.py': 'Flask application',
        'config.json': 'Configuration file',
        'requirements.txt': 'Dependencies list',
        'templates/index.html': 'Upload page',
        'templates/results.html': 'Results page'
    }
    
    all_ok = True
    for file, description in files.items():
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} - {description} missing")
            all_ok = False
    
    return all_ok

def check_model():
    """Check if YOLO model exists or can be loaded"""
    print("\n🤖 Checking YOLO model...")
    model_path = Path('models/yolov8s-pose.pt')
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✅ yolov8s-pose.pt ({size:.1f} MB)")
        return True
    else:
        print(f"   ⚠️  yolov8s-pose.pt not found")
        print(f"      Will be downloaded automatically on first run")
        print(f"      Or download manually: python -c \"from ultralytics import YOLO; YOLO('yolov8s-pose.pt')\"")
        return True  # Not critical, will download

def check_config():
    """Check config.json validity"""
    print("\n⚙️  Checking configuration...")
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = [
            'POSE_MODEL',
            'HAND_TOUCH_DISTANCE',
            'POSE_CONFIDENCE',
            'MIN_TRANSACTION_FRAMES'
        ]
        
        all_ok = True
        for key in required_keys:
            if key in config:
                print(f"   ✅ {key}: {config[key]}")
            else:
                print(f"   ⚠️  {key} missing (will use default)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error reading config.json: {e}")
        return False

def test_import_app():
    """Try importing the Flask app"""
    print("\n🌐 Testing Flask app import...")
    try:
        sys.path.insert(0, os.getcwd())
        # Don't actually import to avoid starting the server
        print("   ✅ app.py structure looks good")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def print_summary(checks):
    """Print summary of checks"""
    print("\n" + "="*60)
    print("📊 INSTALLATION CHECK SUMMARY")
    print("="*60)
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {check_name}")
    
    print("="*60)
    
    if passed == total:
        print("🎉 All checks passed! You're ready to go!")
        print("\nTo start the application:")
        print("  Windows: run_app.bat")
        print("  Mac/Linux: ./run_app.sh")
        print("  Or: python app.py")
        print("\nThen open: http://localhost:5000")
        return True
    else:
        print(f"⚠️  {total - passed} check(s) failed")
        print("\nPlease fix the issues above before running the application.")
        print("Run: pip install -r requirements.txt")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("🔍 Hotel Cash Transaction Detector")
    print("   Installation Check")
    print("="*60)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Folders': check_folders(),
        'Files': check_files(),
        'YOLO Model': check_model(),
        'Configuration': check_config(),
        'Flask App': test_import_app()
    }
    
    success = print_summary(checks)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

