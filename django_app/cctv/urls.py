"""
URL patterns for CCTV app
"""
from django.urls import path
from . import views

app_name = 'cctv'

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main pages
    path('', views.home, name='home'),
    path('monitor/all/', views.monitor_all, name='monitor_all'),
    path('monitor/local/', views.monitor_local, name='monitor_local'),
    path('monitor/local/<int:branch_id>/', views.monitor_local, name='monitor_local_branch'),
    
    # Video management
    path('video/logs/', views.video_logs, name='video_logs'),
    path('video/full/', views.video_full, name='video_full'),
    
    # Branch management
    path('manage/branches/', views.manage_branches, name='manage_branches'),
    path('manage/branches/<int:branch_id>/', views.manage_branch_detail, name='manage_branch_detail'),
    
    # Reports
    path('reports/', views.reports, name='reports'),
    
    # API endpoints
    path('api/branches/', views.api_branches, name='api_branches'),
    path('api/branches/<int:branch_id>/', views.api_branch_detail, name='api_branch_detail'),
    path('api/branches/<int:branch_id>/cameras/', views.api_branch_cameras, name='api_branch_cameras'),
    path('api/cameras/', views.api_cameras, name='api_cameras'),
    path('api/cameras/<int:camera_id>/', views.api_camera_detail, name='api_camera_detail'),
    path('api/events/', views.api_events, name='api_events'),
    path('api/events/<int:event_id>/', views.api_event_detail, name='api_event_detail'),
    path('api/videos/', views.api_videos, name='api_videos'),
    path('api/home-stats/', views.api_home_stats, name='api_home_stats'),
    path('api/report-stats/', views.api_report_stats, name='api_report_stats'),
    
    # Camera management
    path('api/cameras/<int:camera_id>/set-zone/', views.api_set_cashier_zone, name='api_set_cashier_zone'),
    path('api/cameras/<int:camera_id>/toggle-detection/', views.api_toggle_detection, name='api_toggle_detection'),
    
    # Video streaming
    path('video-feed/<int:camera_id>/', views.video_feed, name='video_feed'),
    
    # Branch accounts
    path('api/branches/<int:branch_id>/accounts/', views.api_branch_accounts, name='api_branch_accounts'),
]
