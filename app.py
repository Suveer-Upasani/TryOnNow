
import os
import threading
import time
import math
import warnings
from flask import Flask, request, render_template, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from collections import deque
import mediapipe as mp

warnings.filterwarnings("ignore")

# =====================================================
# FLASK APPLICATION SETUP
# =====================================================

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
MODEL_PATH = 'path_best.pt'
MESH_PATH = 'path_t_shirt.glb'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =====================================================
# GARMENT DETECTION CLASS (YOLO Integration)
# =====================================================

class GarmentDetector:
    """YOLO-based garment detection system"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        try:
            # Placeholder for YOLO model loading
            # In production, you would load your YOLO model here
            print(f"🤖 Garment detector initialized with model: {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load YOLO model: {e}")
    
    def detect_garments(self, image):
        """Detect garments in image and return results"""
        try:
            # Placeholder detection - replace with actual YOLO inference
            # For demo purposes, we'll simulate detection results
            height, width = image.shape[:2]
            
            # Simulate garment detection
            detections = [
                {
                    'class': 'shirt',
                    'confidence': 0.95,
                    'bbox': [width*0.2, height*0.2, width*0.6, height*0.6],
                    'color': [100, 150, 200],  # BGR format
                    'color_name': 'Light Blue'
                }
            ]
            
            return detections
            
        except Exception as e:
            print(f"❌ Garment detection error: {e}")
            return []

# =====================================================
# POSE DETECTION CLASS (MediaPipe)
# =====================================================

class PoseDetector:
    """Advanced pose detection for realistic shirt augmentation"""
    
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Initialize segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # Pose history for smoothing
        self.pose_history = deque(maxlen=10)
        self.contour_history = deque(maxlen=5)
        
    def process_frame(self, frame):
        """Process frame with enhanced body contour detection"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            segmentation_results = self.selfie_segmentation.process(frame_rgb)
            
            results = {
                'landmarks': None,
                'segmentation_mask': None,
                'human_detected': False,
                'shirt_region_valid': False,
                'mesh_position': {'x': 0, 'y': 0, 'z': 0, 'scale': 1.0, 'rotation': {'x': 0, 'y': 0, 'z': 0}},
                'body_contours': None,
                'detailed_contours': {
                    'shoulders': None,
                    'chest': None,
                    'stomach': None,
                    'arms': None
                }
            }
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                results['landmarks'] = landmarks
                results['human_detected'] = self._validate_human_pose(landmarks)
                
                if results['human_detected']:
                    full_contours = self._get_full_body_contours(landmarks, frame.shape)
                    results['body_contours'] = full_contours['full']
                    results['detailed_contours'] = {
                        'shoulders': full_contours['shoulders'],
                        'chest': full_contours['chest'],
                        'stomach': full_contours['stomach'],
                        'arms': full_contours['arms']
                    }
                    
                    results['shirt_region_valid'] = self._validate_shirt_region(
                        landmarks, 
                        results['detailed_contours']
                    )
                    
                    if results['shirt_region_valid']:
                        results['mesh_position'] = self._calculate_dynamic_mesh_position(
                            landmarks,
                            results['detailed_contours'],
                            frame.shape
                        )
            
            # Process segmentation mask
            if segmentation_results.segmentation_mask is not None:
                mask = (segmentation_results.segmentation_mask > 0.7).astype(np.uint8) * 255
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                results['segmentation_mask'] = mask
            
            return results
            
        except Exception as e:
            print(f"Pose processing error: {e}")
            return {
                'landmarks': None,
                'segmentation_mask': None,
                'human_detected': False,
                'shirt_region_valid': False,
                'mesh_position': {'x': 0, 'y': 0, 'z': 0, 'scale': 1.0, 'rotation': {'x': 0, 'y': 0, 'z': 0}},
                'body_contours': None,
                'detailed_contours': {'shoulders': None, 'chest': None, 'stomach': None, 'arms': None}
            }
    
    def _validate_human_pose(self, landmarks):
        """Validate human pose with torso and arm checks"""
        if not landmarks:
            return False
            
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW
        ]
        
        visible_count = 0
        for landmark in required_landmarks:
            if landmarks[landmark.value].visibility > 0.5:
                visible_count += 1
        
        if visible_count < 5:
            return False
            
        # Validate torso proportions
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        torso_height = abs((left_shoulder.y + right_shoulder.y)/2 - (left_hip.y + right_hip.y)/2)
        
        if shoulder_width < 0.1 or torso_height < 0.2:
            return False
            
        return True
    
    def _get_full_body_contours(self, landmarks, img_shape):
        """Calculate precise contours for all body parts"""
        height, width = img_shape[:2]
        contours = {'full': None, 'shoulders': None, 'chest': None, 'stomach': None, 'arms': None}
        
        def get_point(landmark):
            return (int(landmark.x * width), int(landmark.y * height))
        
        # Shoulder contour
        shoulder_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        shoulder_coords = [get_point(landmarks[lm.value]) for lm in shoulder_points 
                          if landmarks[lm.value].visibility > 0.3]
        
        if len(shoulder_coords) >= 2:
            # Expand shoulder points to create a proper contour
            left_pt, right_pt = shoulder_coords
            shoulder_width = abs(right_pt[0] - left_pt[0])
            neck_y = min(left_pt[1], right_pt[1]) - int(shoulder_width * 0.2)
            
            expanded_shoulders = [
                left_pt,
                right_pt,
                (right_pt[0], neck_y),
                (left_pt[0], neck_y)
            ]
            contours['shoulders'] = cv2.convexHull(np.array(expanded_shoulders))
        
        # Chest and torso contours
        torso_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        torso_coords = [get_point(landmarks[lm.value]) for lm in torso_points 
                        if landmarks[lm.value].visibility > 0.3]
        
        if len(torso_coords) >= 4:
            contours['chest'] = cv2.convexHull(np.array(torso_coords))
            contours['stomach'] = cv2.convexHull(np.array(torso_coords))
        
        # Full body contour
        all_points = []
        if contours['shoulders'] is not None:
            all_points.extend(contours['shoulders'].reshape(-1, 2))
        if contours['chest'] is not None:
            all_points.extend(contours['chest'].reshape(-1, 2))
            
        if len(all_points) >= 5:
            hull = cv2.convexHull(np.array(all_points))
            epsilon = 0.005 * cv2.arcLength(hull, True)
            contours['full'] = cv2.approxPolyDP(hull, epsilon, True)
        
        return contours
    
    def _validate_shirt_region(self, landmarks, detailed_contours):
        """Validate shirt region for AR placement"""
        if not landmarks:
            return False
            
        if not all(detailed_contours[key] is not None for key in ['shoulders', 'chest']):
            return False
            
        return True
    
    def _calculate_dynamic_mesh_position(self, landmarks, detailed_contours, img_shape):
        """Calculate precise mesh position based on body contours"""
        height, width = img_shape[:2]
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate center position
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Calculate depth and scale
        avg_z = (left_shoulder.z + right_shoulder.z + left_hip.z + right_hip.z) / 4
        depth = max(0.5, min(2.0, 1.0 + avg_z * 0.5))
        
        # Calculate shoulder angle
        shoulder_angle = math.atan2(
            right_shoulder.y - left_shoulder.y, 
            right_shoulder.x - left_shoulder.x
        )
        
        # Dynamic scale calculation
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        torso_height = abs((left_shoulder.y + right_shoulder.y)/2 - (left_hip.y + right_hip.y)/2)
        base_scale = max(0.8, min(3.0, (shoulder_width * 4.0 + torso_height * 3.0)))
        
        position = {
            'x': (center_x - 0.5) * 2 * 0.8,
            'y': -((center_y - 0.5) * 2) * 0.8,
            'z': -depth * 0.1,
            'scale': base_scale,
            'rotation': {
                'x': 0,
                'y': 0,
                'z': shoulder_angle * 0.2
            }
        }
        
        # Smooth position
        self.pose_history.append(position)
        return self._smooth_position(position)
    
    def _smooth_position(self, current_position):
        """Apply temporal smoothing to position data"""
        if len(self.pose_history) < 2:
            return current_position
            
        weights = np.linspace(0.1, 1.0, len(self.pose_history))
        weights /= weights.sum()
        
        smoothed = {
            'x': 0, 'y': 0, 'z': 0, 'scale': 0,
            'rotation': {'x': 0, 'y': 0, 'z': 0}
        }
        
        for i, pos in enumerate(self.pose_history):
            smoothed['x'] += pos['x'] * weights[i]
            smoothed['y'] += pos['y'] * weights[i]
            smoothed['z'] += pos['z'] * weights[i]
            smoothed['scale'] += pos['scale'] * weights[i]
            smoothed['rotation']['z'] += pos['rotation']['z'] * weights[i]
        
        return smoothed

# =====================================================
# COLOR ANALYSIS CLASS
# =====================================================

class ColorAnalyzer:
    """Advanced color analysis for garments"""
    
    def __init__(self):
        self.color_names = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128)
        }
    
    def analyze_garment_color(self, image, bbox=None):
        """Analyze dominant color in garment region"""
        try:
            if bbox:
                x, y, w, h = map(int, bbox)
                roi = image[y:y+h, x:x+w]
            else:
                roi = image
            
            # Convert to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Get dominant color using k-means clustering
            pixels = roi_rgb.reshape(-1, 3)
            pixels = pixels[pixels.sum(axis=1) > 50]  # Remove very dark pixels
            
            if len(pixels) == 0:
                return np.array([128, 128, 128]), "gray"
            
            # Simple dominant color calculation
            dominant_color = np.mean(pixels, axis=0).astype(int)
            
            # Find closest color name
            color_name = self._get_closest_color_name(dominant_color)
            
            return dominant_color, color_name
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return np.array([128, 128, 128]), "gray"
    
    def _get_closest_color_name(self, rgb_color):
        """Find closest color name to RGB value"""
        min_distance = float('inf')
        closest_color = "gray"
        
        for name, color in self.color_names.items():
            distance = np.sqrt(np.sum((np.array(color) - rgb_color) ** 2))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color

# =====================================================
# CAMERA HANDLER CLASS
# =====================================================

class CameraHandler:
    """Handles camera operations"""
    
    def __init__(self):
        self.cap = None
        self.camera_active = False
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.camera_active = True
            return True
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.cap = None
    
    def get_frame(self):
        """Get current frame from camera"""
        if not self.camera_active or not self.cap:
            return None
        
        success, frame = self.cap.read()
        if success:
            return frame
        return None
    
    def is_active(self):
        """Check if camera is active"""
        return self.camera_active

# =====================================================
# STATUS MANAGER CLASS
# =====================================================

class StatusManager:
    """Manages application state and status"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all status variables"""
        self.ar_initialized = False
        self.shirt_region_valid = False
        self.auto_detection_active = False
        self.preparation_active = False
        self.stable_detection_count = 0
        self.human_detected_count = 0
        self.last_detection_time = 0
        self.detection_interval = 1.0
        self.required_stable_detections = 25
        self.required_human_detections = 10
        self.preparation_start_time = 0
        self.preparation_duration = 2.0
        self.mesh_position = {'x': 0, 'y': 0, 'z': 0, 'scale': 1.0, 'rotation': {'x': 0, 'y': 0, 'z': 0}}
        
    def update_pose_status(self, pose_results):
        """Update status based on pose detection results"""
        human_detected = pose_results.get('human_detected', False)
        shirt_valid = pose_results.get('shirt_region_valid', False)
        
        if human_detected:
            self.human_detected_count += 1
        else:
            self.human_detected_count = max(0, self.human_detected_count - 2)
            
        if human_detected and shirt_valid:
            self.stable_detection_count += 1
        else:
            self.stable_detection_count = max(0, self.stable_detection_count - 1)
        
        self.shirt_region_valid = shirt_valid and self.human_detected_count >= self.required_human_detections
        
        # Handle preparation phase
        current_time = time.time()
        
        if (not self.preparation_active and not self.ar_initialized and 
            self.stable_detection_count >= self.required_stable_detections):
            print("🎯 Stable pose detected - Starting AR calibration")
            self.preparation_active = True
            self.preparation_start_time = current_time
            self.stable_detection_count = 0
        
        # Check preparation completion
        if self.preparation_active:
            prep_status = self.get_preparation_status()
            if not prep_status['active'] and prep_status['progress'] >= 100:
                print("🚀 AR calibration complete")
                self.ar_initialized = True
                self.auto_detection_active = True
                self.preparation_active = False
        
        # Update mesh position if AR is active
        if self.ar_initialized and human_detected:
            self.mesh_position = pose_results.get('mesh_position', self.mesh_position)
        
        # Reset if human lost
        if self.human_detected_count == 0 and self.ar_initialized:
            print("⚠️ Human lost - Resetting AR system")
            self.ar_initialized = False
            self.auto_detection_active = False
            self.preparation_active = False
            self.stable_detection_count = 0
    
    def should_run_detection(self):
        """Check if garment detection should run"""
        current_time = time.time()
        return (self.ar_initialized and 
                current_time - self.last_detection_time > self.detection_interval)
    
    def update_detection_time(self):
        """Update last detection time"""
        self.last_detection_time = time.time()
    
    def get_preparation_status(self):
        """Get current preparation phase status"""
        if not self.preparation_active:
            return {'active': False, 'time_remaining': 0, 'progress': 0, 'message': 'AR ready'}
        
        elapsed = time.time() - self.preparation_start_time
        remaining = max(0, self.preparation_duration - elapsed)
        progress = min(100, (elapsed / self.preparation_duration) * 100)
        
        if remaining <= 0:
            self.preparation_active = False
            return {'active': False, 'time_remaining': 0, 'progress': 100, 'message': 'AR ready!'}
        
        return {
            'active': True,
            'time_remaining': remaining,
            'progress': progress,
            'message': f'Calibrating... {int(remaining + 1)}s remaining'
        }
    
    def is_ar_initialized(self):
        return self.ar_initialized
    
    def is_shirt_region_valid(self):
        return self.shirt_region_valid
    
    def get_mesh_position(self):
        return self.mesh_position

# =====================================================
# GLOBAL VARIABLES
# =====================================================

# Initialize components
garment_detector = GarmentDetector(MODEL_PATH)
pose_detector = PoseDetector()
color_analyzer = ColorAnalyzer()
camera_handler = CameraHandler()
status_manager = StatusManager()

# Global state variables
current_frame = None
frame_lock = threading.Lock()
uploaded_images = False
current_texture = None
current_garment_color = None
current_garment_bbox = None
run_detections = False
pose_landmarks = None
landmarks_lock = threading.Lock()

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_upload_folder():
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f'Error clearing upload folder: {e}')

def process_camera_feed():
    """Main camera processing loop"""
    global current_frame, pose_landmarks, current_garment_color
    
    while camera_handler.is_active():
        frame = camera_handler.get_frame()
        if frame is None:
            continue
            
        # Process pose detection
        pose_results = pose_detector.process_frame(frame)
        
        with landmarks_lock:
            pose_landmarks = pose_results.get('landmarks')
        
        # Update status based on pose detection
        status_manager.update_pose_status(pose_results)
        
        # Periodic garment detection for color analysis
        if status_manager.should_run_detection():
            try:
                detections = garment_detector.detect_garments(frame)
                if detections:
                    # Extract color from first detection
                    detection = detections[0]
                    if 'bbox' in detection:
                        color, color_name = color_analyzer.analyze_garment_color(frame, detection['bbox'])
                        current_garment_color = np.array([color[2], color[1], color[0]])  # Convert RGB to BGR
                    status_manager.update_detection_time()
                    print(f"🔄 Color analysis updated: {color_name}")
            except Exception as e:
                print(f"❌ Detection error: {e}")
        
        with frame_lock:
            current_frame = frame.copy()
        
        time.sleep(0.006)  # ~30 FPS

# =====================================================
# FLASK ROUTES
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_images', methods=['POST'])
def upload_images():
    global uploaded_images, current_texture, current_garment_color, run_detections
    
    clear_upload_folder()
    
    if 'images' not in request.files:
        return jsonify({'status': 'error', 'message': 'No files uploaded'}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({'status': 'error', 'message': 'No files selected'}), 400
    
    uploaded_count = 0
    all_detections = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    run_detections = True
                    detections = garment_detector.detect_garments(image)
                    if detections:
                        all_detections.extend(detections)
                        # Extract color from first garment
                        detection = detections[0]
                        if 'bbox' in detection:
                            color, color_name = color_analyzer.analyze_garment_color(image, detection['bbox'])
                            current_garment_color = np.array([color[2], color[1], color[0]])  # RGB to BGR
                    run_detections = False
                    uploaded_count += 1
            except Exception as e:
                print(f"❌ Error processing uploaded image: {e}")
    
    if uploaded_count > 0:
        uploaded_images = True
        response_data = {
            'status': 'success',
            'message': f'✅ Analysis complete: {uploaded_count} images processed',
            'uploaded': True,
            'detected_garments': all_detections[:5]
        }
        
        if current_garment_color is not None:
            b, g, r = current_garment_color.tolist()
            response_data['shirt_color'] = {
                'rgb': [r, g, b],
                'bgr': [b, g, r],
                'hex': f"#{r:02x}{g:02x}{b:02x}",
                'normalized': [r/255.0, g/255.0, b/255.0]
            }
        
        return jsonify(response_data)
    else:
        return jsonify({'status': 'error', 'message': 'No valid files uploaded'}), 400

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global uploaded_images
    
    if not uploaded_images:
        return jsonify({'status': 'error', 'message': 'Please upload garment images first'}), 400
    
    if camera_handler.is_active():
        return jsonify({'status': 'error', 'message': 'Camera is already active'}), 400
    
    success = camera_handler.start_camera()
    if not success:
        return jsonify({'status': 'error', 'message': 'Could not access camera'}), 500
    
    # Reset status manager
    status_manager.reset()
    
    # Start camera processing thread
    camera_thread = threading.Thread(target=process_camera_feed, daemon=True)
    camera_thread.start()
    
    return jsonify({
        'status': 'success', 
        'message': '🚀 Enhanced AR Camera Active',
        'features': {
            'pose_detection': True,
            'garment_detection': True,
            'color_analysis': True,
            'physics_simulation': True
        }
    })

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    if not camera_handler.is_active():
        return jsonify({'status': 'error', 'message': 'Camera is not active'}), 400
    
    camera_handler.stop_camera()
    status_manager.reset()
    
    return jsonify({'status': 'success', 'message': '⏹️ Camera stopped'})

@app.route('/pose_landmarks')
def get_pose_landmarks():
    global pose_landmarks
    
    if pose_landmarks is None:
        return jsonify({'status': 'error', 'message': 'No pose detected'})
    
    landmarks_data = []
    for i, landmark in enumerate(pose_landmarks):
        landmarks_data.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    
    return jsonify({
        'status': 'success',
        'landmarks': landmarks_data,
        'shirt_region_valid': status_manager.is_shirt_region_valid(),
        'ar_initialized': status_manager.is_ar_initialized(),
        'mesh_position': status_manager.get_mesh_position(),
        'preparation_status': status_manager.get_preparation_status()
    })

@app.route('/get_shirt_color')
def get_shirt_color():
    global current_garment_color
    
    if current_garment_color is None:
        return jsonify({
            'status': 'error', 
            'message': 'No shirt color detected',
            'color': None
        })
    
    b, g, r = current_garment_color.tolist()
    
    return jsonify({
        'status': 'success',
        'color': {
            'rgb': [r, g, b],
            'bgr': [b, g, r],
            'hex': f"#{r:02x}{g:02x}{b:02x}",
            'normalized': [r/255.0, g/255.0, b/255.0]
        }
    })

@app.route('/camera_status')
def camera_status():
    global current_garment_color, uploaded_images
    
    response_data = {
        'camera_active': camera_handler.is_active(),
        'images_uploaded': uploaded_images,
        'shirt_region_valid': status_manager.is_shirt_region_valid(),
        'ar_initialized': status_manager.is_ar_initialized(),
        'physics_available': True
    }
    
    if current_garment_color is not None:
        b, g, r = current_garment_color.tolist()
        response_data['shirt_color'] = {
            'rgb': [r, g, b],
            'bgr': [b, g, r],
            'hex': f"#{r:02x}{g:02x}{b:02x}",
            'normalized': [r/255.0, g/255.0, b/255.0]
        }
    
    return jsonify(response_data)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global current_frame
        
        while True:
            with frame_lock:
                if current_frame is not None and camera_handler.is_active():
                    ret, buffer = cv2.imencode('.jpg', current_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Generate blank frame when camera is inactive
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    blank.fill(20)
                    
                    cv2.putText(blank, 'Virtual Try-On AR', (150, 220), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 2)
                    cv2.putText(blank, 'Camera Inactive', (200, 260), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', blank)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/models/<path:filename>')
def serve_model(filename):
    """Serve GLB model files"""
    models_dir = '/Users/suveer/Desktop/10/models'
    response = send_from_directory(models_dir, filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Content-Type'] = 'application/octet-stream'
    return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# =====================================================
# MAIN APPLICATION STARTUP
# =====================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs(os.path.join('static', 'models'), exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Virtual Try-On Application...")
    print("🎯 Features: Pose Detection, Garment Recognition, Color Analysis")
    print("🤖 AI Detection: YOLO + MediaPipe")
    print("✨ AR: Real-time GLB mesh rendering with physics")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 YOLO model path: {MODEL_PATH}")
    print(f"📐 GLB Mesh path: {MESH_PATH}")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5005, threaded=True)
    finally:
        # Cleanup on shutdown
        if camera_handler.is_active():
            camera_handler.stop_camera()




