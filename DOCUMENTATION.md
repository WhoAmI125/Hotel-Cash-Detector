# 호텔 행동 감지 시스템

---

## 1. 시스템 개요

### 1.1 목적
호텔 행동 감지 시스템은 호텔 카운터(Cashier) 감시를 위한 AI 기반 CCTV 모니터링 시스템입니다. 다음 이벤트를 자동 감지합니다:
- 현금 거래 - 캐셔와 고객 간의 (hand-to-hand) 현금 전달
- 폭력/난동 - 물리적 충돌 또는 공격적인 행동
- 화재/연기 - 화재 및 연기 감지(안전용)

### 1.2 주요 특징
- RTSP 실시간 비디오 스트림 처리
- 카메라별 개별 설정이 가능한 다중 카메라 지원
- 백그라운드 감지 워커(지속 모니터링)
- 이벤트 로그 및 비디오 클립 녹화
- 다국어 지원 (영어, 한국어, 태국어, 베트남어, 중국어)
- 역할 기반 접근 제어 (관리자, 프로젝트 매니저)
- 디버깅 및 튜닝을 위한 개발자 모드 제공

---

## 2. 기술 스택

### 2.1 백엔드 프레임워크

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Web Framework | Django | 5.2.7 | Main web application, ORM, admin |
| Python | Python | 3.10+ | Core programming language |
| ASGI Server | Daphne/Uvicorn | Latest | Async request handling |
| Task Queue | Threading | Built-in | Background detection workers |

### 2.2 프론트엔드

| Component | Technology | Purpose |
|-----------|------------|---------|
| Templates | Django Templates | Server-side rendering |
| Styling | Custom CSS | Dark theme UI |
| JavaScript | Vanilla JS | Interactive components |
| Video | HTML5 Video + MJPEG | Live stream display |

### 2.3 AI/ML 스택

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Deep Learning | PyTorch | 2.0+ | Model inference backend |
| Object Detection | Ultralytics YOLOv8 | 8.0+ | Person, fire detection |
| Pose Estimation | YOLOv8-Pose | 8.0+ | Hand position tracking |
| Computer Vision | OpenCV | 4.8+ | Frame processing, video I/O |
| Array Operations | NumPy | 1.24+ | Numerical computations |

### 2.4 데이터베이스

| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | SQLite3 | Development database |
| ORM | Django ORM | Database abstraction |
| Migrations | Django Migrations | Schema versioning |

### 2.5 비디오 처리

| Component | Technology | Purpose |
|-----------|------------|---------|
| Stream Protocol | RTSP over TCP | Camera connection |
| Video Codec | H.264 (libx264) | Clip encoding |
| Transcoding | FFmpeg | Video conversion |
| Container | MP4 (faststart) | Web-compatible video |

---

## 3. 아키텍처

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        클라이언트 레이어                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   브라우저  │  │   모바일    │  │   관리자 대시보드       │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DJANGO 웹 레이어                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Views    │  │     API     │  │   비디오 스트리밍       │  │
│  │   (HTML)    │  │   (JSON)    │  │   (MJPEG/MP4)           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   백그라운드 워커 레이어                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │       BackgroundCameraWorker (카메라별 1개)                 ││
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐││
│  │  │  RTSP    │─▶│   Unified    │─▶│   이벤트/클립 저장     │││
│  │  │ 캡처     │  │  Detector    │  │   (비동기)             │││
│  │  └──────────┘  └──────────────┘  └────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    감지 레이어                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    현금     │  │   폭력      │  │         화재           │  │
│  │  감지기     │  │  감지기     │  │       감지기           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          ▼                                       │
│                   ┌─────────────┐                                │
│                   │ YOLOv8 +    │                                │
│                   │ YOLOv8-Pose │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    데이터 레이어                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   SQLite    │  │   미디어    │  │       모델들           │  │
│  │  Database   │  │   파일      │  │   (YOLO 가중치)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

```

### 3.2 디렉터리 구조

```
Hotel-Cash-Detector/
├── django_app/                    # 메인 Django 애플리케이션
│   ├── manage.py                  # Django 관리 스크립트
│   ├── db.sqlite3                 # SQLite 데이터베이스
│   ├── hotel_cctv/                # Django 프로젝트 설정
│   │   ├── settings.py            # 설정
│   │   ├── urls.py                # 루트 URL 라우팅
│   │   ├── wsgi.py                # WSGI 엔트리 포인트
│   │   └── asgi.py                # ASGI 엔트리 포인트
│   ├── cctv/                      # 메인 앱
│   │   ├── models.py              # 데이터베이스 모델
│   │   ├── views.py               # 뷰 및 API 엔드포인트
│   │   ├── urls.py                # 앱 URL 라우팅
│   │   ├── admin.py               # 관리자 설정
│   │   ├── translations.py        # 다국어 지원
│   │   └── context_processors.py  # 템플릿 컨텍스트
│   ├── templates/cctv/            # HTML 템플릿
│   │   ├── base.html              # 기본 템플릿
│   │   ├── home.html              # 대시보드
│   │   ├── monitor_all.html       # 다중 카메라 화면
│   │   ├── monitor_local.html     # 단일 카메라 화면
│   │   ├── camera_settings.html   # 카메라 설정
│   │   ├── video_logs.html        # 이벤트 로그
│   │   └── ...
│   ├── static/                    # 정적 파일
│   │   ├── css/style.css          # 스타일
│   │   └── js/main.js             # JavaScript
│   ├── media/                     # 업로드/생성 파일
│   │   ├── clips/                 # 이벤트 비디오 클립
│   │   └── thumbnails/            # 이벤트 썸네일
│   └── models/                    # AI 모델 가중치
│       ├── yolov8s.pt             # YOLOv8 Small (사람 감지)
│       ├── yolov8s-pose.pt        # YOLOv8 Pose (손 추적)
│       └── fire_smoke_yolov8.pt   # 화재/연기 감지
│
└── detectors/                     # 감지 모듈 (루트 레벨)
    ├── base_detector.py           # 베이스 클래스
    ├── unified_detector.py        # 메인 통합 감지기
    ├── cash_detector.py           # 현금 감지
    ├── violence_detector.py       # 폭력 감지
    └── fire_detector.py           # 화재 감지

```

---

## 4. Database Schema

### 4.1 Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│      User       │       │     Region      │       │     Branch      │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │
│ username        │       │ name            │       │ name            │
│ email           │       │ code            │       │ region_id (FK)  │
│ password        │       └────────┬────────┘       │ address         │
│ role            │                │                │ status          │
│ phone           │                │                │ created_at      │
└────────┬────────┘                │                └────────┬────────┘
         │                         │                         │
         │    ┌────────────────────┘                         │
         │    │                                              │
         ▼    ▼                                              ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│   managers (M2M)        │                    │        Camera           │
│   Branch ←──────────────┼────────────────────├─────────────────────────┤
│        → User           │                    │ id (PK)                 │
└─────────────────────────┘                    │ branch_id (FK)          │
                                               │ camera_id               │
                                               │ name                    │
                                               │ rtsp_url                │
                                               │ status                  │
                                               │ cashier_zone_*          │
                                               │ cash_confidence         │
                                               │ violence_confidence     │
                                               │ fire_confidence         │
                                               │ hand_touch_distance     │
                                               │ detect_cash             │
                                               │ detect_violence         │
                                               │ detect_fire             │
                                               └───────────┬─────────────┘
                                                           │
                                                           ▼
                                               ┌─────────────────────────┐
                                               │         Event           │
                                               ├─────────────────────────┤
                                               │ id (PK)                 │
                                               │ branch_id (FK)          │
                                               │ camera_id (FK)          │
                                               │ event_type              │
                                               │ status                  │
                                               │ confidence              │
                                               │ frame_number            │
                                               │ bbox_*                  │
                                               │ clip_path               │
                                               │ thumbnail_path          │
                                               │ notes                   │
                                               │ reviewed_by (FK)        │
                                               │ created_at              │
                                               └─────────────────────────┘
```

### 4.2 모델 정의

#### User 모델
```python
class User(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Admin (Master)'),
        ('project_manager', 'Project Manager'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    phone = models.CharField(max_length=20, blank=True, null=True)
```

| 필드         | 타입         | 설명                           |
| ---------- | ---------- | ---------------------------- |
| `id`       | AutoField  | 기본 키                         |
| `username` | CharField  | 로그인 아이디                      |
| `email`    | EmailField | 이메일 주소                       |
| `password` | CharField  | 해시된 비밀번호                     |
| `role`     | CharField  | `admin` 또는 `project_manager` |
| `phone`    | CharField  | 전화번호(선택)                     |


#### Region 모델
```python
class Region(models.Model):
    name = models.CharField(max_length=50, unique=True)
    code = models.CharField(max_length=10, unique=True)
```

| 필드     | 타입        | 설명                   |
| ------ | --------- | -------------------- |
| `id`   | AutoField | 기본 키                 |
| `name` | CharField | 지역 이름 (예: "Bangkok") |
| `code` | CharField | 지역 코드 (예: "BKK")     |

#### Branch 모델
```python
class Branch(models.Model):
    name = models.CharField(max_length=100)
    region = models.ForeignKey(Region, on_delete=models.CASCADE)
    address = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    managers = models.ManyToManyField(User, related_name='managed_branches')
```

| 필드          | 타입         | 설명                                  |
| ----------- | ---------- | ----------------------------------- |
| `id`        | AutoField  | 기본 키                                |
| `name`      | CharField  | 지점(호텔/브랜치) 이름                       |
| `region_id` | ForeignKey | Region 참조                           |
| `address`   | TextField  | 지점 주소                               |
| `status`    | CharField  | `confirmed`, `reviewing`, `pending` |
| `managers`  | ManyToMany | 담당 프로젝트 매니저들                        |


#### Camera 모델
```python
class Camera(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    camera_id = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    
    # Cashier zone coordinates
    cashier_zone_x = models.IntegerField(default=0)
    cashier_zone_y = models.IntegerField(default=0)
    cashier_zone_width = models.IntegerField(default=640)
    cashier_zone_height = models.IntegerField(default=480)
    cashier_zone_enabled = models.BooleanField(default=False)
    
    # Detection thresholds
    cash_confidence = models.FloatField(default=0.5)
    violence_confidence = models.FloatField(default=0.6)
    fire_confidence = models.FloatField(default=0.5)
    hand_touch_distance = models.IntegerField(default=100)
    
    # Detection toggles
    detect_cash = models.BooleanField(default=True)
    detect_violence = models.BooleanField(default=True)
    detect_fire = models.BooleanField(default=True)
```

| 필드                    | 타입         | 설명                                 |
| --------------------- | ---------- | ---------------------------------- |
| `id`                  | AutoField  | 기본 키                               |
| `branch_id`           | ForeignKey | Branch 참조                          |
| `camera_id`           | CharField  | 카메라 식별자(외부 ID)                     |
| `name`                | CharField  | 표시용 카메라 이름                         |
| `rtsp_url`            | CharField  | RTSP 스트림 URL                       |
| `status`              | CharField  | `online`, `offline`, `maintenance` |
| `cashier_zone_*`      | Integer    | 캐셔 존 좌표 (x, y, width, height)      |
| `cash_confidence`     | Float      | 현금 감지 임곗값 (0.0-1.0)                |
| `violence_confidence` | Float      | 폭력 감지 임곗값                          |
| `fire_confidence`     | Float      | 화재 감지 임곗값                          |
| `hand_touch_distance` | Integer    | 손-손 간 최대 픽셀 거리                     |


#### Event 모델
```python
class Event(models.Model):
    TYPE_CHOICES = [
        ('cash', '현금'),
        ('fire', '화재'),
        ('violence', '난동'),
    ]
    
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    confidence = models.FloatField(default=0.0)
    clip_path = models.CharField(max_length=500, blank=True, null=True)
    thumbnail_path = models.CharField(max_length=500, blank=True, null=True)
```

| 필드               | 타입         | 설명                                  |
| ---------------- | ---------- | ----------------------------------- |
| `id`             | AutoField  | 기본 키                                |
| `branch_id`      | ForeignKey | Branch 참조                           |
| `camera_id`      | ForeignKey | Camera 참조                           |
| `event_type`     | CharField  | `cash`, `violence`, `fire`          |
| `status`         | CharField  | `confirmed`, `reviewing`, `pending` |
| `confidence`     | Float      | 감지 신뢰도 (0.0-1.0)                    |
| `frame_number`   | Integer    | 감지 시점 프레임 번호                        |
| `bbox_*`         | Integer    | 바운딩 박스 좌표                           |
| `clip_path`      | CharField  | 비디오 클립 경로                           |
| `thumbnail_path` | CharField  | 썸네일 이미지 경로                          |
| `reviewed_by`    | ForeignKey | 검수한 사용자                             |
| `created_at`     | DateTime   | 이벤트 발생 시각                           |


---

## 5. 감지 모델

### 5.1 모델 개요

| 모델           | 파일                     | 크기     | 용도                  |
| ------------ | ---------------------- | ------ | ------------------- |
| YOLOv8n      | `yolov8n.pt`           | ~6 MB  | 사람 감지(백업)           |
| YOLOv8n-Pose | `yolov8n-pose.pt`      | ~7 MB  | 포즈 추정 (17 키포인트)     |
| YOLOv8s-Pose | `yolov8s-pose.pt`      | ~23 MB | 더 높은 정확도의 포즈 추정(옵션) |
| Fire/Smoke   | `fire_smoke_yolov8.pt` | ~6 MB  | 화재 및 연기 감지          |


### 5.2 현금 거래 감지

알고리즘: 캐셔-고객 구분 + 포즈 기반 손-손 거리 계산

```
┌────────────────────────────────────────────────────────────┐
│                 현금 감지 파이프라인                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. 프레임 입력                                             │
│     └── RTSP 스트림 프레임 (일반적으로 1920x1080)           │
│                                                            │
│  2. 포즈 추정 (YOLOv8-Pose)                                │
│     └── 프레임 내 모든 사람 감지                           │
│     └── 사람당 17개 키포인트 추출                          │
│     └── 손목 키포인트 집중: LEFT_WRIST(9), RIGHT_WRIST(10) │
│     └── 키포인트 최소 신뢰도: pose_confidence (기본 0.5)   │
│                                                            │
│  3. 사람 중심점 계산                                        │
│     └── 우선: 엉덩이 중심(가장 안정적)                     │
│         └── keypoints[11] (left_hip) + keypoints[12]       │
│     └── 대안: 어깨 중심                                    │
│         └── keypoints[5] (left_shoulder) + keypoints[6]    │
│     └── 최후: 바운딩 박스 중심                             │
│         └── center = ((x1+x2)/2, (y1+y2)/2)                │
│                                                            │
│  4. 존 분류 (엄격 모드)                                    │
│     └── 바운딩 박스가 아닌 "중심점"만 사용                 │
│     └── 중심점이 캐셔 존 안에 있으면 → CASHIER(캐셔)      │
│     └── 중심점이 존 밖에 있으면 → CUSTOMER(고객)          │
│     └── 한 사람은 반드시 하나의 존에만 속하도록 보장      │
│                                                            │
│  5. 손 위치 추출                                           │
│     └── 사람별:                                             │
│         ├── left_wrist = keypoints[9]                      │
│         ├── right_wrist = keypoints[10]                    │
│         └── 키포인트 신뢰도 ≥ 0.3인 경우에만 사용          │
│                                                            │
│  6. 손-손 근접도 검사 (엄격 XOR 검증)                      │
│     └── 각 사람 쌍(i, j)에 대해:                           │
│         ├── 둘 다 IN 존이면 스킵 (캐셔-캐셔)               │
│         ├── 둘 다 OUT 존이면 스킵 (고객-고객)              │
│         ├── XOR 조건만 허용: (p1_in XOR p2_in)             │
│         │   └── 정확히 한 명은 존 안, 한 명은 존 밖        │
│         │                                                  │
│         └── 유효한 캐셔-고객 쌍에 대해서:                  │
│             └── 네 가지 손 조합 모두 검사:                 │
│                 ├── cashier.left ↔ customer.left           │
│                 ├── cashier.left ↔ customer.right          │
│                 ├── cashier.right ↔ customer.left          │
│                 └── cashier.right ↔ customer.right         │
│                 └── distance = √((x1-x2)² + (y1-y2)²)      │
│                                                            │
│  7. 감지 조건 (모두 만족해야 함)                           │
│     ✓ 한 사람의 중심점이 캐셔 존 안에 있음                 │
│     ✓ 한 사람의 중심점이 캐셔 존 밖에 있음                 │
│     ✓ 두 사람 모두 손 키포인트를 가짐 (신뢰도 ≥ 0.3)       │
│     ✓ 손-손 거리 < hand_touch_distance                      │
│     ✓ distance score = 1 - (distance/threshold)            │
│     ✓ consecutive_detections ≥ min_transaction_frames      │
│     ✓ transaction_cooldown(쿨다운 프레임 수) 경과           │
│                                                            │
│  8. 메타데이터 수집                                         │
│     └── 캐셔 정보:                                          │
│         ├── center: [x, y]                                 │
│         ├── bbox: [x1, y1, x2, y2]                         │
│         ├── hands: {left: [x,y,conf], right: [x,y,conf]}  │
│         ├── in_zone: true                                  │
│         └── hand_used: "left" 또는 "right"                 │
│     └── 고객 정보:                                          │
│         ├── center: [x, y]                                 │
│         ├── bbox: [x1, y1, x2, y2]                         │
│         ├── hands: {left: [x,y,conf], right: [x,y,conf]}  │
│         ├── in_zone: false                                 │
│         └── hand_used: "left" 또는 "right"                 │
│     └── 감지 정보:                                          │
│         ├── distance: 실제 손-손 픽셀 거리                  │
│         ├── distance_threshold: 사용된 임곗값               │
│         ├── interaction_point: 손 중간 지점 [x, y]         │
│         └── people_count: 프레임 내 전체 사람 수           │
│                                                            │
│  9. 이벤트 생성                                             │
│     └── 메타데이터를 포함한 Detection 객체 생성            │
│     └── 클립 저장 트리거(비동기, 30초 버퍼)                │
│     └── 전체 메타데이터를 가진 JSON 파일 저장              │
│     └── 데이터베이스 Event 레코드 생성                     │
│     └── 감지 프레임 기준 썸네일 생성                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

```

#### 핵심 로직 개선 사항

**1. 바운딩 박스(Bbox) 대신 중심점 기반 존 분류**
```python
# OLD: Bbox overlap could split one person into two zones
if self.is_box_in_cashier_zone(bbox, threshold=0.3):
    # Problem: 30% overlap = ambiguous

# NEW: Single center point = definitive classification
center = self.get_person_center(keypoints, bbox)  # Hip or shoulder center
if self.is_in_cashier_zone(center):
    # One person = one zone = no ambiguity
```

**2. 엄격한 XOR 검증**
```python
# Enforce cashier-customer pairs ONLY
p1_in = person1['in_cashier_zone']
p2_in = person2['in_cashier_zone']
is_valid_pair = (p1_in and not p2_in) or (not p1_in and p2_in)

if not is_valid_pair:
    continue  # Skip: both cashiers OR both customers
```

**3. 상세 메타데이터 기록**
- 캐셔 위치(중심점, 바운딩 박스, 손 좌표)
- 고객 위치(중심점, 바운딩 박스, 손 좌표)
- 실제 측정된 손-손 거리
- 감지를 트리거한 임곗값
- 상호작용 지점(손 중간점)
- JSON에 모두 저장하여 분석 및 튜닝에 활용

**주요 파라미터:**
- `hand_touch_distance`: Maximum pixel distance between hands (default: 100px)
- `pose_confidence`: Minimum keypoint confidence (default: 0.3)
- `min_transaction_frames`: Frames before confirming (default: 1)
- `transaction_cooldown`: Frames between detections (default: 45)

**Keypoint Indices (COCO Format):**
```
0: nose          5: left_shoulder   10: right_wrist
1: left_eye      6: right_shoulder  11: left_hip
2: right_eye     7: left_elbow      12: right_hip
3: left_ear      8: right_elbow     13: left_knee
4: right_ear     9: left_wrist      14: right_knee
                                    15: left_ankle
                                    16: right_ankle

```

### 5.3 폭력 감지

**알고리즘:** 포즈 기반 근접 몸싸움 감지

```
┌────────────────────────────────────────────────────────────┐
│               폭력(난동) 감지 파이프라인                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. 포즈 추정                                              │
│     └── 프레임 내 모든 사람에 대해 포즈(키포인트) 추정     │
│                                                            │
│  2. 근접 전투(Close Combat) 후보 탐지                      │
│     └── 서로 근접한 사람 쌍 탐색                           │
│     └── 바운딩 박스가 겹치는지 확인                        │
│     └── 사람 사이의 거리 계산                              │
│                                                            │
│  3. 공격적인 포즈 분석                                     │
│     └── 팔이 위로 올라가 있거나 휘두르는 모양인지 확인     │
│     └── 프레임 간 빠른 움직임(모션) 검출                   │
│     └── 양쪽 모두 공격성 지표를 보일 때만 후보 인정       │
│                                                            │
│  4. 지속성 기준 적용                                       │
│     └── min_violence_frames 이상 연속 프레임에서 발생해야  │
│     └── violence_confidence 임곗값 이상일 것               │
│                                                            │
│  5. 예외 처리                                               │
│     └── 캐셔 존 내 일반 거래(손 터치 등)는 폭력에서 제외   │
│     └── 한 명만 있는 경우 단독 행동은 폭력으로 보지 않음  │
│                                                            │
└────────────────────────────────────────────────────────────┘

```

**주요 파라미터:**
- `violence_confidence`: Detection threshold (default: 0.8)
- `min_violence_frames`: Consecutive frames required (default: 15)
- `motion_threshold`: Motion magnitude threshold (default: 100)
- `violence_cooldown`: Frames between alerts (default: 90)

### 5.4 화재 감지

**알고리즘:** YOLO + 색상 기반 감지 + 깜빡임 분석

```
┌────────────────────────────────────────────────────────────┐
│                 화재 감지 파이프라인                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1차 방법: YOLO 화재/연기 모델                              │
│  ├── 클래스: {0: 'Fire', 1: 'default', 2: 'smoke'}         │
│  ├── conf=0.25로 추론 수행                                 │
│  └── fire_confidence 임곗값 이상인 결과만 사용             │
│                                                            │
│  2차(폴백) 방법: 색상 기반 감지                            │
│  ├── 프레임을 HSV 색공간으로 변환                          │
│  ├── 불꽃 색역: 밝은 주황/노랑 (H:5-25, S:150+)           │
│  ├── 피부톤 영역 제외(오탐 방지)                           │
│  ├── 시간에 따른 밝기 변화(깜빡임) 분석                   │
│  └── 충분한 영역 + 깜빡임 점수를 동시에 만족해야 함       │
│                                                            │
│  연기 감지:                                                │
│  ├── 배경 차분(Background Subtraction, MOG2)              │
│  ├── 회색/흰색 영역 마스크                                │
│  └── 위로 이동하는 모션(연기 상승) 검출                    │
│                                                            │
│  최종 확인:                                                │
│  ├── min_fire_frames 이상 연속 감지                        │
│  ├── camera별 fire_confidence 임곗값 만족                  │
│  └── fire_cooldown 경과 후에만 재알림                     │
│                                                            │
└────────────────────────────────────────────────────────────┘

```

**주요 파라미터:**
- `fire_confidence`: Detection threshold (default: 0.7)
- `min_fire_frames`: Consecutive frames required (default: 10)
- `min_fire_area`: Minimum fire region area (default: 3000 px²)
- `fire_cooldown`: Frames between alerts (default: 60)

**색상 범위 (HSV):**
```python
# Fire colors (very bright orange/yellow)
fire_lower1 = [5, 150, 200]    fire_upper1 = [25, 255, 255]
fire_lower2 = [0, 200, 220]    fire_upper2 = [5, 255, 255]

# Skin exclusion (to prevent false positives)
skin_lower = [0, 20, 70]       skin_upper = [25, 170, 200]

# Smoke (gray/white)
smoke_lower = [0, 0, 150]      smoke_upper = [180, 30, 255]
```



## 6. 테스트 & 검증

### 6.1 테스트 스크립트

시스템에는 통합 테스트 스크립트 test_worker_streaming.py 가 포함되어 있습니다.

```bash
python test_worker_streaming.py
```

#### Test Coverage

| 테스트                 | 설명                | 성공 기준                   |
| ------------------- | ----------------- | ----------------------- |
| **Camera Model**    | DB 연결 및 카메라 설정 조회 | 유효한 RTSP URL을 가진 카메라 조회 |
| **RTSP Connection** | 스트림 연결 테스트        | 10초 이내 연결, 유효한 해상도      |
| **Frame Reading**   | 스트림 안정성           | 모든 프레임 읽기 성공, FPS > 20  |
| **Detector Init**   | 모델 로딩             | 모든 감지기 초기화 성공           |
| **Detection**       | 프레임 감지            | 오류 없이 프레임 처리            |
| **Cash Metadata**   | 메타데이터 구조          | 예상 키가 모두 존재             |
| **JSON Output**     | 파일 생성             | 유효한 JSON 구조 생성          |
| **Event Model**     | DB 연산             | 이벤트 생성/조회 가능            |


#### 샘플 출력

```
╔════════════════════════════════════════════════════════════╗
║     WORKER & STREAMING TEST SUITE                         ║
║     Hotel Cash Detector System                            ║
╚════════════════════════════════════════════════════════════╝

============================================================
  Test 1: Camera Model
============================================================

  ✅ PASS: Found 1 camera(s) in database
  ℹ️ INFO: Camera ID: 1
  ℹ️ INFO: Camera Name: test
  ℹ️ INFO: RTSP URL: rtsp://admin:adminadmin!@175.213.55.16:554
  ℹ️ INFO: Hand Touch Distance: 200px
  ℹ️ INFO: Cashier Zone: x=2, y=354, w=1273, h=359

============================================================
  Test 2: RTSP Stream Connection
============================================================

  ℹ️ INFO: Connecting to: rtsp://admin:adminadmin!@...
  ✅ PASS: Stream opened in 4.0s
  ℹ️ INFO: Stream FPS: 25.0
  ℹ️ INFO: Resolution: 1280x720

============================================================
  Test 3: Frame Reading
============================================================

  ℹ️ INFO: Reading 30 frames...
  ✅ PASS: Read all 30/30 frames successfully
  ℹ️ INFO: Actual FPS: 108.4
  ℹ️ INFO: Frame shape: (720, 1280, 3)

============================================================
  TEST SUMMARY
============================================================

  ✅ PASS: camera_model
  ✅ PASS: rtsp_connection
  ✅ PASS: frame_reading
  ✅ PASS: detector_init
  ✅ PASS: detection
  ✅ PASS: cash_metadata
  ✅ PASS: json_output
  ✅ PASS: event_model

  Results: 8/8 tests passed
  Time: 6.2s

  ✅ ALL TESTS PASSED!
```
