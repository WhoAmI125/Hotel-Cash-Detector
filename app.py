"""
Flask App for Hotel Cash Transaction Detection
Upload videos and extract clips around detected transactions
Uses the EXACT same detection logic as main.py
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import json
import threading
import uuid
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np
import math
import time

# Import the detection classes from main.py
from main import SimpleHandTouchConfig, SimpleHandTouchDetector

app = Flask(__name__)

# Load configuration from config.json
CONFIG_FILE = 'config.json'
try:
    with open(CONFIG_FILE, 'r') as f:
        APP_CONFIG = json.load(f)
    print(f"✅ Loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"⚠️  Config file not found, using defaults")
    APP_CONFIG = {
        'MAX_FILE_SIZE_MB': 500,
        'ALLOWED_EXTENSIONS': ['mp4', 'avi', 'mov', 'mkv'],
        'MAX_VIDEOS_PER_UPLOAD': 5,
        'SECONDS_BEFORE_TRANSACTION': 2,
        'SECONDS_AFTER_TRANSACTION': 2
    }

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG.get('MAX_FILE_SIZE_MB', 500) * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = set(APP_CONFIG.get('ALLOWED_EXTENSIONS', ['mp4', 'avi', 'mov', 'mkv']))
app.config['MAX_VIDEOS'] = APP_CONFIG.get('MAX_VIDEOS_PER_UPLOAD', 5)

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

# Store processing status
processing_status = {}

print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
print(f"📏 Max file size: {APP_CONFIG.get('MAX_FILE_SIZE_MB', 500)}MB")
print(f"🎥 Max videos per upload: {app.config['MAX_VIDEOS']}")
print()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class TransactionClipExtractor:
    """Extract clips around detected transactions using the EXACT logic from main.py"""
    
    def __init__(self, config_dict=None):
        # Create SimpleHandTouchConfig from dict
        self.config = SimpleHandTouchConfig()
        
        if config_dict:
            # Update config with provided values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Initialize the detector with our config
        self.detector = SimpleHandTouchDetector(self.config)
        
        # Clip extraction settings from config
        self.SECONDS_BEFORE = config_dict.get('SECONDS_BEFORE_TRANSACTION', 2)
        self.SECONDS_AFTER = config_dict.get('SECONDS_AFTER_TRANSACTION', 2)
        
        print(f"🤖 Model: {self.config.POSE_MODEL}")
        print(f"👋 Hand touch distance: {self.config.HAND_TOUCH_DISTANCE}px")
        print(f"⏱️  Clip padding: {self.SECONDS_BEFORE}s before, {self.SECONDS_AFTER}s after")
        if self.config.CASHIER_ZONE:
            print(f"🎯 Cashier Zone: {self.config.CASHIER_ZONE}")
        print()
    
    def detect_transactions(self, video_path, progress_callback=None):
        """Detect all transactions in video using SimpleHandTouchDetector"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reset detector state for new video
        self.detector.transaction_history = {}
        self.detector.person_id_map = {}
        self.detector.next_stable_id = 1
        self.detector.cashier_persistence = {}
        self.detector.stats = {
            'frames': 0,
            'transactions': 0,
            'confirmed_transactions': 0
        }
        
        transactions = []
        current_transaction = None
        frame_num = 0
        
        print(f"📹 Analyzing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Progress update
            if progress_callback and frame_num % 30 == 0:
                progress_callback(int(frame_num / total_frames * 100))
            
            # Use the EXACT same detection method from main.py
            output_frame, confirmed_transactions = self.detector.detect_hand_touches(frame)
            
            # Track transactions
            if confirmed_transactions:
                for trans in confirmed_transactions:
                    # Check if this is a new transaction or continuation
                    trans_key = f"{trans['p1_id']}-{trans['p2_id']}"
                    
                    if current_transaction is None or current_transaction['key'] != trans_key:
                        # New transaction detected
                        if current_transaction:
                            # Save previous transaction
                            transactions.append(current_transaction)
                        
                        current_transaction = {
                            'key': trans_key,
                            'start_frame': frame_num,
                            'end_frame': frame_num,
                            'start_time': frame_num / fps,
                            'end_time': frame_num / fps,
                            'p1_id': trans['p1_id'],
                            'p2_id': trans['p2_id'],
                            'hand_type': trans['hand_type']
                        }
                    else:
                        # Continue existing transaction
                        current_transaction['end_frame'] = frame_num
                        current_transaction['end_time'] = frame_num / fps
            else:
                # No transactions in this frame
                if current_transaction:
                    # End the current transaction
                    transactions.append(current_transaction)
                    current_transaction = None
        
        # Add last transaction if exists
        if current_transaction:
            transactions.append(current_transaction)
        
        cap.release()
        
        # Merge close transactions
        merged_transactions = self._merge_close_transactions(transactions, fps)
        
        print(f"✅ Detection complete: Found {len(merged_transactions)} transaction(s)")
        
        return merged_transactions, fps
    
    def _merge_close_transactions(self, transactions, fps):
        """Merge transactions that are close together"""
        if not transactions:
            return []
        
        merged = []
        current = transactions[0].copy()
        
        for trans in transactions[1:]:
            # If transactions are within 1 second and same pair, merge them
            if (trans['start_time'] - current['end_time'] <= 1.0 and 
                trans['key'] == current['key']):
                current['end_frame'] = trans['end_frame']
                current['end_time'] = trans['end_time']
            else:
                merged.append(current)
                current = trans.copy()
        
        merged.append(current)
        return merged
    
    def extract_clips(self, video_path, transactions, fps, output_folder, progress_callback=None):
        """Extract video clips around transactions"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        clips = []
        
        for idx, trans in enumerate(transactions):
            # Calculate clip boundaries
            start_frame = max(0, trans['start_frame'] - int(self.SECONDS_BEFORE * fps))
            end_frame = trans['end_frame'] + int(self.SECONDS_AFTER * fps)
            
            # Create output path
            video_name = Path(video_path).stem
            clip_name = f"{video_name}_transaction_{idx+1}_P{trans['p1_id']}_P{trans['p2_id']}_{int(trans['start_time'])}s.mp4"
            clip_path = os.path.join(output_folder, clip_name)
            
            # Extract clip
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            clips.append({
                'filename': clip_name,
                'path': clip_path,
                'start_time': round(trans['start_time'], 2),
                'end_time': round(trans['end_time'], 2),
                'duration': round(trans['end_time'] - trans['start_time'], 2),
                'p1_id': trans['p1_id'],
                'p2_id': trans['p2_id'],
                'hand_type': trans.get('hand_type', 'N/A'),
                'frames': frames_written
            })
            
            if progress_callback:
                progress_callback(int((idx + 1) / len(transactions) * 100))
            
            print(f"  📎 Clip {idx+1}: {clip_name} ({frames_written} frames)")
        
        cap.release()
        return clips


def process_video(job_id, video_path, video_filename):
    """Process video in background thread"""
    try:
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'stage': 'Initializing detector...',
            'filename': video_filename,
            'started': datetime.now().isoformat()
        }
        
        # Create extractor with full config
        extractor = TransactionClipExtractor(APP_CONFIG)
        
        # Detect transactions
        processing_status[job_id]['stage'] = 'Detecting transactions...'
        
        def detection_progress(progress):
            processing_status[job_id]['progress'] = progress // 2  # First 50%
        
        transactions, fps = extractor.detect_transactions(video_path, detection_progress)
        
        processing_status[job_id]['stage'] = f'Found {len(transactions)} transaction(s). Extracting clips...'
        processing_status[job_id]['transactions_count'] = len(transactions)
        
        if len(transactions) == 0:
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['clips'] = []
            processing_status[job_id]['message'] = 'No transactions detected'
            return
        
        # Extract clips
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        Path(output_folder).mkdir(exist_ok=True)
        
        def extraction_progress(progress):
            processing_status[job_id]['progress'] = 50 + (progress // 2)  # Last 50%
        
        clips = extractor.extract_clips(video_path, transactions, fps, output_folder, extraction_progress)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['clips'] = clips
        processing_status[job_id]['completed'] = datetime.now().isoformat()
        
        print(f"✅ Job {job_id} completed: {len(clips)} clips extracted")
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['error'] = str(e)
        processing_status[job_id]['progress'] = 0
        print(f"❌ Job {job_id} error: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle multiple video uploads"""
    if 'videos' not in request.files:
        return jsonify({'error': 'No videos provided'}), 400
    
    files = request.files.getlist('videos')
    
    max_videos = app.config.get('MAX_VIDEOS', 5)
    if len(files) > max_videos:
        return jsonify({'error': f'Maximum {max_videos} videos allowed'}), 400
    
    jobs = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            job_id = str(uuid.uuid4())
            
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(upload_path)
            
            # Start processing in background
            thread = threading.Thread(target=process_video, args=(job_id, upload_path, filename))
            thread.daemon = True
            thread.start()
            
            jobs.append({
                'job_id': job_id,
                'filename': filename
            })
    
    return jsonify({'jobs': jobs})


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])


@app.route('/download/<job_id>/<filename>')
def download_clip(job_id, filename):
    """Download extracted clip"""
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    return send_from_directory(output_folder, filename, as_attachment=True)


@app.route('/results/<job_id>')
def view_results(job_id):
    """View results page"""
    if job_id not in processing_status:
        return "Job not found", 404
    
    return render_template('results.html', job_id=job_id, status=processing_status[job_id])


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Flask Cash Transaction Detector")
    print("="*70)
    print("Using EXACT same detection logic as main.py")
    print("Open your browser: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
