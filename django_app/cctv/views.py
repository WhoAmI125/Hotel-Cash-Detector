"""
Views for Hotel CCTV Monitoring System
"""
import json
import cv2
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Q
from django.conf import settings

from .models import User, Region, Branch, Camera, Event, VideoRecord, BranchAccount

# Add flask directory to path for detector imports
FLASK_DIR = Path(settings.BASE_DIR).parent / 'flask'
sys.path.insert(0, str(FLASK_DIR))

# Try to import detectors
try:
    from detectors import UnifiedDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    print("Warning: Detectors not available")

# Global detector instances per camera
camera_detectors = {}


def get_user_branches(user):
    """Get branches accessible by the user"""
    if user.is_admin():
        return Branch.objects.all()
    return user.managed_branches.all()


# ==================== AUTHENTICATION ====================

def login_view(request):
    """Login page"""
    if request.user.is_authenticated:
        return redirect('cctv:home')
    
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', '/')
            return redirect(next_url)
        else:
            error = '아이디 또는 비밀번호가 올바르지 않습니다.'
    
    return render(request, 'cctv/login.html', {'error': error})


def logout_view(request):
    """Logout"""
    logout(request)
    return redirect('cctv:login')


# ==================== MAIN PAGES ====================

@login_required
def home(request):
    """Dashboard home page"""
    user = request.user
    branches = get_user_branches(user)
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'home',
    }
    return render(request, 'cctv/home.html', context)


@login_required
def monitor_all(request):
    """All branches monitoring page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:monitor_local')
    
    branches = Branch.objects.all().select_related('region')
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'monitor-all',
    }
    return render(request, 'cctv/monitor_all.html', context)


@login_required
def monitor_local(request, branch_id=None):
    """Local branch monitoring page"""
    user = request.user
    user_branches = get_user_branches(user)
    
    if branch_id:
        branch = get_object_or_404(Branch, id=branch_id)
        if not user.is_admin() and branch not in user_branches:
            return redirect('cctv:home')
    else:
        branch = user_branches.first()
    
    cameras = branch.cameras.all() if branch else []
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branch': branch,
        'branches': user_branches,
        'cameras': cameras,
        'regions': regions,
        'active_page': 'monitor-local',
    }
    return render(request, 'cctv/monitor_local.html', context)


@login_required
def video_logs(request):
    """Event logs page"""
    user = request.user
    user_branches = get_user_branches(user)
    regions = Region.objects.all()
    
    # Get filter parameters
    date_filter = request.GET.get('date', timezone.now().date().isoformat())
    region_filter = request.GET.get('region', 'all')
    type_filter = request.GET.get('type', 'all')
    branch_filter = request.GET.get('branch', '')
    
    # Build query
    events = Event.objects.select_related('branch', 'camera', 'branch__region')
    
    if not user.is_admin():
        events = events.filter(branch__in=user_branches)
    
    if date_filter:
        events = events.filter(created_at__date=date_filter)
    
    if region_filter != 'all':
        events = events.filter(branch__region__name=region_filter)
    
    if type_filter != 'all':
        events = events.filter(event_type=type_filter)
    
    if branch_filter:
        events = events.filter(branch__name__icontains=branch_filter)
    
    events = events.order_by('-created_at')[:100]
    
    # Get offline cameras
    offline_cameras = Camera.objects.filter(status='offline').select_related('branch')
    if not user.is_admin():
        offline_cameras = offline_cameras.filter(branch__in=user_branches)
    
    context = {
        'user': user,
        'events': events,
        'offline_cameras': offline_cameras,
        'regions': regions,
        'active_page': 'video-logs',
        'filters': {
            'date': date_filter,
            'region': region_filter,
            'type': type_filter,
            'branch': branch_filter,
        }
    }
    return render(request, 'cctv/video_logs.html', context)


@login_required
def video_full(request):
    """Full videos page"""
    user = request.user
    user_branches = get_user_branches(user)
    regions = Region.objects.all()
    
    # Get filter parameters
    date_filter = request.GET.get('date', '')
    region_filter = request.GET.get('region', 'all')
    branch_filter = request.GET.get('branch', '')
    
    # Build query
    videos = VideoRecord.objects.select_related('branch', 'camera', 'branch__region')
    
    if not user.is_admin():
        videos = videos.filter(branch__in=user_branches)
    
    if date_filter:
        videos = videos.filter(recorded_date=date_filter)
    
    if region_filter != 'all':
        videos = videos.filter(branch__region__name=region_filter)
    
    if branch_filter:
        videos = videos.filter(branch__name__icontains=branch_filter)
    
    videos = videos.order_by('-recorded_date')[:50]
    
    context = {
        'user': user,
        'videos': videos,
        'regions': regions,
        'active_page': 'video-full',
        'filters': {
            'date': date_filter,
            'region': region_filter,
            'branch': branch_filter,
        }
    }
    return render(request, 'cctv/video_full.html', context)


@login_required
def manage_branches(request):
    """Branch management page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:home')
    
    branches = Branch.objects.all().select_related('region')
    regions = Region.objects.all()
    
    # Get filter parameters
    region_filter = request.GET.get('region', '전체')
    search_filter = request.GET.get('search', '')
    
    if region_filter != '전체':
        branches = branches.filter(region__name=region_filter)
    
    if search_filter:
        branches = branches.filter(name__icontains=search_filter)
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'manage-branches',
        'filters': {
            'region': region_filter,
            'search': search_filter,
        }
    }
    return render(request, 'cctv/manage_branches.html', context)


@login_required
def manage_branch_detail(request, branch_id):
    """Branch detail management page"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return redirect('cctv:home')
    
    accounts = branch.accounts.all()
    cameras = branch.cameras.all()
    
    context = {
        'user': user,
        'branch': branch,
        'accounts': accounts,
        'cameras': cameras,
        'active_page': 'manage-branch-detail',
    }
    return render(request, 'cctv/manage_branch_detail.html', context)


@login_required
def reports(request):
    """Reports page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:home')
    
    regions = Region.objects.all()
    branches = Branch.objects.all().select_related('region')
    
    context = {
        'user': user,
        'regions': regions,
        'branches': branches,
        'active_page': 'reports',
    }
    return render(request, 'cctv/reports.html', context)


# ==================== API ENDPOINTS ====================

@login_required
@require_http_methods(["GET", "POST"])
def api_branches(request):
    """API for branches"""
    user = request.user
    
    if request.method == 'GET':
        branches = get_user_branches(user).select_related('region')
        data = [{
            'id': b.id,
            'name': b.name,
            'region': b.region.name,
            'status': b.status,
            'status_display': b.get_status_display(),
            'camera_count': b.get_camera_count(),
            'event_count': b.get_today_event_count(),
        } for b in branches]
        return JsonResponse({'branches': data})
    
    elif request.method == 'POST':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        data = json.loads(request.body)
        region = get_object_or_404(Region, name=data.get('region'))
        
        branch = Branch.objects.create(
            name=data.get('name'),
            region=region,
            status='pending'
        )
        
        return JsonResponse({
            'success': True,
            'branch': {
                'id': branch.id,
                'name': branch.name,
                'region': branch.region.name,
            }
        })


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_branch_detail(request, branch_id):
    """API for single branch"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': branch.id,
            'name': branch.name,
            'region': branch.region.name,
            'status': branch.status,
            'status_display': branch.get_status_display(),
            'address': branch.address,
            'camera_count': branch.get_camera_count(),
        })
    
    elif request.method == 'DELETE':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        branch.delete()
        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
@require_http_methods(["GET", "POST"])
def api_branch_cameras(request, branch_id):
    """API for branch cameras"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        cameras = branch.cameras.all()
        data = [{
            'id': c.id,
            'camera_id': c.camera_id,
            'name': c.name,
            'location': c.location,
            'status': c.status,
            'status_display': c.get_status_display(),
            'rtsp_url': c.rtsp_url,
        } for c in cameras]
        return JsonResponse({'cameras': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        
        camera = Camera.objects.create(
            branch=branch,
            camera_id=data.get('camera_id'),
            name=data.get('name'),
            location=data.get('location', ''),
            rtsp_url=data.get('rtsp_url'),
            status='offline'
        )
        
        return JsonResponse({
            'success': True,
            'camera': {
                'id': camera.id,
                'camera_id': camera.camera_id,
                'name': camera.name,
            }
        })


@login_required
@require_http_methods(["GET", "POST"])
def api_cameras(request):
    """API for all cameras"""
    user = request.user
    user_branches = get_user_branches(user)
    
    if request.method == 'GET':
        cameras = Camera.objects.filter(branch__in=user_branches).select_related('branch')
        data = [{
            'id': c.id,
            'camera_id': c.camera_id,
            'name': c.name,
            'branch': c.branch.name,
            'branch_id': c.branch.id,
            'location': c.location,
            'status': c.status,
            'status_display': c.get_status_display(),
        } for c in cameras]
        return JsonResponse({'cameras': data})


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_camera_detail(request, camera_id):
    """API for single camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': camera.id,
            'camera_id': camera.camera_id,
            'name': camera.name,
            'branch': camera.branch.name,
            'location': camera.location,
            'status': camera.status,
            'rtsp_url': camera.rtsp_url,
            'cashier_zone': camera.get_cashier_zone(),
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
        })
    
    elif request.method == 'PUT':
        data = json.loads(request.body)
        camera.name = data.get('name', camera.name)
        camera.location = data.get('location', camera.location)
        camera.rtsp_url = data.get('rtsp_url', camera.rtsp_url)
        camera.save()
        return JsonResponse({'success': True})
    
    elif request.method == 'DELETE':
        camera.delete()
        return JsonResponse({'success': True})


@login_required
def api_events(request):
    """API for events"""
    user = request.user
    user_branches = get_user_branches(user)
    
    # Get filter parameters
    date_filter = request.GET.get('date')
    region_filter = request.GET.get('region')
    type_filter = request.GET.get('type')
    branch_filter = request.GET.get('branch')
    limit = int(request.GET.get('limit', 50))
    
    events = Event.objects.filter(branch__in=user_branches).select_related('branch', 'camera', 'branch__region')
    
    if date_filter:
        events = events.filter(created_at__date=date_filter)
    
    if region_filter and region_filter != 'all':
        events = events.filter(branch__region__name=region_filter)
    
    if type_filter and type_filter != 'all':
        events = events.filter(event_type=type_filter)
    
    if branch_filter:
        events = events.filter(branch__name__icontains=branch_filter)
    
    events = events.order_by('-created_at')[:limit]
    
    data = [{
        'id': e.id,
        'branch': e.branch.name,
        'region': e.branch.region.name,
        'camera': e.camera.camera_id,
        'type': e.event_type,
        'type_display': e.get_event_type_display(),
        'status': e.status,
        'status_display': e.get_status_display(),
        'confidence': e.confidence,
        'date': e.created_at.strftime('%Y-%m-%d'),
        'time': e.created_at.strftime('%H:%M'),
        'created_at': e.created_at.isoformat(),
    } for e in events]
    
    return JsonResponse({'events': data})


@login_required
@require_http_methods(["GET", "PUT"])
def api_event_detail(request, event_id):
    """API for single event"""
    user = request.user
    event = get_object_or_404(Event, id=event_id)
    
    if not user.is_admin() and event.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': event.id,
            'branch': event.branch.name,
            'camera': event.camera.camera_id,
            'type': event.event_type,
            'type_display': event.get_event_type_display(),
            'status': event.status,
            'status_display': event.get_status_display(),
            'confidence': event.confidence,
            'frame_number': event.frame_number,
            'bbox': event.get_bbox(),
            'clip_path': event.clip_path,
            'created_at': event.created_at.isoformat(),
        })
    
    elif request.method == 'PUT':
        data = json.loads(request.body)
        event.status = data.get('status', event.status)
        event.notes = data.get('notes', event.notes)
        if data.get('status') in ['confirmed', 'reviewing']:
            event.reviewed_by = user
            event.reviewed_at = timezone.now()
        event.save()
        return JsonResponse({'success': True})


@login_required
def api_videos(request):
    """API for video records"""
    user = request.user
    user_branches = get_user_branches(user)
    
    videos = VideoRecord.objects.filter(branch__in=user_branches).select_related('branch', 'camera', 'branch__region')
    
    date_filter = request.GET.get('date')
    region_filter = request.GET.get('region')
    branch_filter = request.GET.get('branch')
    
    if date_filter:
        videos = videos.filter(recorded_date=date_filter)
    
    if region_filter and region_filter != 'all':
        videos = videos.filter(branch__region__name=region_filter)
    
    if branch_filter:
        videos = videos.filter(branch__name__icontains=branch_filter)
    
    videos = videos.order_by('-recorded_date')[:50]
    
    data = [{
        'id': v.id,
        'branch': v.branch.name,
        'region': v.branch.region.name,
        'camera': v.camera.camera_id,
        'file_id': v.file_id,
        'date': v.recorded_date.isoformat(),
        'duration': v.duration,
        'file_size': v.file_size,
    } for v in videos]
    
    return JsonResponse({'videos': data})


@login_required
def api_home_stats(request):
    """API for home page statistics"""
    user = request.user
    user_branches = get_user_branches(user)
    today = timezone.now().date()
    
    stats = []
    for branch in user_branches:
        event_count = branch.events.filter(created_at__date=today).count()
        confirmed = branch.events.filter(created_at__date=today, status='confirmed').count()
        pending = branch.events.filter(created_at__date=today, status='pending').count()
        
        if pending > 0:
            status = 'pending'
        elif confirmed == event_count and event_count > 0:
            status = 'confirmed'
        else:
            status = 'reviewing'
        
        stats.append({
            'branch': branch.name,
            'count': event_count,
            'status': status,
        })
    
    return JsonResponse({'stats': stats})


@login_required
def api_report_stats(request):
    """API for report statistics"""
    user = request.user
    if not user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    user_branches = get_user_branches(user)
    today = timezone.now().date()
    month_start = today.replace(day=1)
    
    # Monthly stats
    monthly_events = Event.objects.filter(
        branch__in=user_branches,
        created_at__date__gte=month_start
    ).count()
    
    # Event type breakdown
    type_counts = Event.objects.filter(
        branch__in=user_branches,
        created_at__date__gte=month_start
    ).values('event_type').annotate(count=Count('id'))
    
    type_breakdown = {t['event_type']: t['count'] for t in type_counts}
    total = sum(type_breakdown.values()) or 1
    
    pie_data = [
        {'label': '현금', 'value': round(type_breakdown.get('cash', 0) / total * 100), 'color': '#1c1373'},
        {'label': '화재', 'value': round(type_breakdown.get('fire', 0) / total * 100), 'color': '#f28c28'},
        {'label': '난동', 'value': round(type_breakdown.get('violence', 0) / total * 100), 'color': '#3cb371'},
    ]
    
    # Daily stats for last 7 days
    daily_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Event.objects.filter(
            branch__in=user_branches,
            created_at__date=day
        ).count()
        daily_data.append({
            'x': day.strftime('%m-%d'),
            'y': count
        })
    
    # Branch summary
    branch_summary = []
    for branch in user_branches[:5]:
        total_events = branch.events.filter(created_at__date__gte=month_start).count()
        branch_summary.append({
            'branch': branch.name,
            'total': total_events,
            'avg': '02:30',  # Placeholder
            'falseRate': '3.0%',  # Placeholder
        })
    
    return JsonResponse({
        'monthly_events': monthly_events,
        'pie_data': pie_data,
        'daily_data': daily_data,
        'branch_summary': branch_summary,
    })


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_set_cashier_zone(request, camera_id):
    """Set cashier zone for a camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    zone = data.get('zone')
    
    if zone and len(zone) == 4:
        camera.set_cashier_zone(zone)
        return JsonResponse({'success': True, 'zone': zone})
    
    return JsonResponse({'error': 'Invalid zone format'}, status=400)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_toggle_detection(request, camera_id):
    """Toggle detection types for a camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    detection_type = data.get('type')
    enabled = data.get('enabled', True)
    
    if detection_type == 'cash':
        camera.detect_cash = enabled
    elif detection_type == 'violence':
        camera.detect_violence = enabled
    elif detection_type == 'fire':
        camera.detect_fire = enabled
    
    camera.save()
    
    return JsonResponse({
        'success': True,
        'detect_cash': camera.detect_cash,
        'detect_violence': camera.detect_violence,
        'detect_fire': camera.detect_fire,
    })


@login_required
@require_http_methods(["GET", "POST"])
def api_branch_accounts(request, branch_id):
    """API for branch accounts"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        accounts = branch.accounts.all()
        data = [{
            'id': a.id,
            'name': a.name,
            'email': a.email,
            'role': a.role,
            'role_display': a.get_role_display(),
        } for a in accounts]
        return JsonResponse({'accounts': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        account = BranchAccount.objects.create(
            branch=branch,
            name=data.get('name'),
            email=data.get('email'),
            role=data.get('role', 'staff')
        )
        return JsonResponse({
            'success': True,
            'account': {
                'id': account.id,
                'name': account.name,
                'email': account.email,
                'role': account.role,
            }
        })


# ==================== VIDEO STREAMING ====================

def get_detector_for_camera(camera):
    """Get or create detector for a camera"""
    global camera_detectors
    
    if not DETECTOR_AVAILABLE:
        return None
    
    if camera.id not in camera_detectors:
        config = {
            'models_dir': str(settings.DETECTION_CONFIG['MODELS_DIR']),
            'cashier_zone': camera.get_cashier_zone(),
            'hand_touch_distance': settings.DETECTION_CONFIG['HAND_TOUCH_DISTANCE'],
            'pose_confidence': settings.DETECTION_CONFIG['POSE_CONFIDENCE'],
            'min_transaction_frames': settings.DETECTION_CONFIG['MIN_TRANSACTION_FRAMES'],
            'fire_confidence': settings.DETECTION_CONFIG['FIRE_CONFIDENCE'],
            'min_fire_frames': settings.DETECTION_CONFIG['MIN_FIRE_FRAMES'],
            'min_fire_area': settings.DETECTION_CONFIG['MIN_FIRE_AREA'],
            'violence_confidence': settings.DETECTION_CONFIG['VIOLENCE_CONFIDENCE'],
            'min_violence_frames': settings.DETECTION_CONFIG['VIOLENCE_DURATION'],
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
        }
        camera_detectors[camera.id] = UnifiedDetector(config)
    
    return camera_detectors[camera.id]


def generate_frames(camera):
    """Generator for video frames with detection"""
    cap = cv2.VideoCapture(camera.rtsp_url)
    
    if not cap.isOpened():
        # Return placeholder frame
        frame = create_placeholder_frame("Cannot connect to camera")
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # Update camera status
    camera.status = 'online'
    camera.last_connected = timezone.now()
    camera.save()
    
    detector = get_detector_for_camera(camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = settings.DETECTION_CONFIG['FRAME_SKIP']
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame with detector
            if detector:
                result = detector.process_frame(frame, draw_overlay=True)
                frame = result['frame']
                
                # Save detections to database
                for det in result.get('detections', []):
                    save_detection(camera, det, frame_count)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        cap.release()
        camera.status = 'offline'
        camera.save()


def create_placeholder_frame(text):
    """Create a placeholder frame with text"""
    frame = cv2.zeros((480, 640, 3), dtype='uint8')
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def save_detection(camera, detection, frame_number):
    """Save detection to database"""
    Event.objects.create(
        branch=camera.branch,
        camera=camera,
        event_type=detection.get('label', 'unknown').lower(),
        confidence=detection.get('confidence', 0),
        frame_number=frame_number,
        bbox_x1=detection.get('bbox', [0, 0, 0, 0])[0],
        bbox_y1=detection.get('bbox', [0, 0, 0, 0])[1],
        bbox_x2=detection.get('bbox', [0, 0, 0, 0])[2],
        bbox_y2=detection.get('bbox', [0, 0, 0, 0])[3],
    )


@login_required
def video_feed(request, camera_id):
    """Video streaming endpoint"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    user = request.user
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    return StreamingHttpResponse(
        generate_frames(camera),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
