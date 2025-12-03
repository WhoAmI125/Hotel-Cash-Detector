"""
Django Admin configuration for CCTV app
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Region, Branch, Camera, Event, VideoRecord, BranchAccount


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'is_staff', 'is_active']
    list_filter = ['role', 'is_staff', 'is_active']
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Role', {'fields': ('role', 'phone')}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('Role', {'fields': ('role', 'phone')}),
    )


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    list_display = ['name', 'code']
    search_fields = ['name', 'code']


@admin.register(Branch)
class BranchAdmin(admin.ModelAdmin):
    list_display = ['name', 'region', 'status', 'created_at']
    list_filter = ['region', 'status']
    search_fields = ['name']
    filter_horizontal = ['managers']


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['camera_id', 'name', 'branch', 'status', 'location']
    list_filter = ['branch', 'status']
    search_fields = ['camera_id', 'name']


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ['id', 'branch', 'camera', 'event_type', 'status', 'created_at']
    list_filter = ['event_type', 'status', 'branch']
    search_fields = ['camera__camera_id', 'branch__name']
    date_hierarchy = 'created_at'


@admin.register(VideoRecord)
class VideoRecordAdmin(admin.ModelAdmin):
    list_display = ['file_id', 'branch', 'camera', 'recorded_date']
    list_filter = ['branch', 'recorded_date']
    search_fields = ['file_id', 'branch__name']


@admin.register(BranchAccount)
class BranchAccountAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'branch', 'role']
    list_filter = ['branch', 'role']
    search_fields = ['name', 'email']
