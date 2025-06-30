import cv2
import numpy as np
import mediapipe as mp
import pyrender
import trimesh
from collections import deque
import time
from scipy import ndimage
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List

class VirtualTryOnSystem:
    def __init__(self):
        # Hardcoded paths and parameters
        self.SHIRT_MESH_PATH = "/Users/suveer/Desktop/11/models/t_shirt.glb"
        self.CAMERA_ID = 0
        self.FRAME_WIDTH = 1280
        self.FRAME_HEIGHT = 720
        self.TARGET_FPS = 30
        
        # Initialize models with optimized parameters
        self._initialize_models()
        
        # Visualization settings
        self._setup_visualization()
        
        # Performance optimization
        self._setup_performance()
        
        # State management
        self._setup_state()
        
        # 3D rendering setup
        self._setup_3d_rendering()
    
    def _initialize_models(self):
        """Initialize all ML models with optimized parameters"""
        # MediaPipe pose estimation with balanced model complexity
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balanced model
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # MediaPipe selfie segmentation for fast person detection
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie.SelfieSegmentation(
            model_selection=1  # General model
        )
    
    def _setup_visualization(self):
        """Configure visualization parameters"""
        # Color schemes
        self.shirt_color = np.array([0, 255, 100], dtype=np.uint8)  # Bright green
        self.outline_color = (0, 255, 255)  # Yellow outline
        self.detail_color = (255, 100, 0)   # Orange details
        
        # Display toggles
        self.show_mask = True
        self.show_outline = True
        self.show_details = True
        self.show_3d = True
        self.debug_mode = False
    
    def _setup_performance(self):
        """Configure performance optimization parameters"""
        # Frame processing
        self.frame_skip = 2  # Process every nth frame for pose detection
        self.frame_count = 0
        
        # Smoothing and tracking
        self.mask_history = deque(maxlen=5)
        self.pose_history = deque(maxlen=3)
        self.prev_shirt_mask = None
        self.prev_pose_landmarks = None
        
        # Color-based segmentation
        self.color_tolerance = 35
        self.min_shirt_area = 5000
        
        # FPS tracking
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def _setup_state(self):
        """Initialize system state variables"""
        # Movement tracking
        self.movement_history = deque(maxlen=10)
        self.stabilization_factor = 0.9
        
        # First frame flag
        self.first_frame_processed = False
    
    def _setup_3d_rendering(self):
        """Initialize 3D rendering components"""
        # 3D shirt properties
        self.shirt_mesh = self._load_shirt_mesh()
        self.renderer = None
        self.scene = None
        self.camera = None
        self.shirt_node = None
        self.shirt_scale = 1.0
        self.shirt_position = np.array([0.0, 0.0, 0.0])
        self.shirt_rotation = np.eye(3)
    
    def _load_shirt_mesh(self) -> Optional[trimesh.Trimesh]:
        """Load and prepare the 3D shirt mesh"""
        try:
            # Load the mesh with error handling
            if not os.path.exists(self.SHIRT_MESH_PATH):
                print(f"Error: Shirt mesh file not found at {self.SHIRT_MESH_PATH}")
                return None
                
            scene_or_mesh = trimesh.load(self.SHIRT_MESH_PATH)
            
            # Handle both Scene and Mesh objects
            if isinstance(scene_or_mesh, trimesh.Scene):
                mesh = next(iter(scene_or_mesh.geometry.values()))
            else:
                mesh = scene_or_mesh
                
            if not isinstance(mesh, trimesh.Trimesh):
                print("Error: Invalid mesh type")
                return None
                
            # Prepare the mesh
            mesh.apply_translation(-mesh.centroid)  # Center the mesh
            
            # Scale down if too large
            max_dim = max(mesh.extents)
            if max_dim > 10:
                mesh.apply_scale(10/max_dim)
            
            print("3D shirt mesh loaded successfully")
            return mesh
        except Exception as e:
            print(f"Error loading shirt mesh: {e}")
            return None
    
    def _initialize_renderer(self, frame_width: int, frame_height: int):
        """Set up the 3D renderer for shirt augmentation"""
        if self.shirt_mesh is None:
            return
            
        try:
            # Create offscreen renderer
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=frame_width,
                viewport_height=frame_height,
                point_size=1.0
            )
            
            # Configure scene with lighting
            self.scene = pyrender.Scene(
                ambient_light=[0.5, 0.5, 0.5],
                bg_color=[0, 0, 0, 0]  # Transparent background
            )
            
            # Create shirt material with slight transparency
            shirt_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.3, 0.3, 0.8, 0.9],
                metallicFactor=0.1,
                roughnessFactor=0.7,
                alphaMode='BLEND'
            )
            
            # Add shirt to scene
            self.shirt_node = pyrender.Mesh.from_trimesh(
                self.shirt_mesh,
                material=shirt_material,
                smooth=True
            )
            self.scene.add(self.shirt_node)
            
            # Set up orthographic camera
            self.camera = pyrender.OrthographicCamera(
                xmag=frame_width/2,
                ymag=frame_height/2,
                zfar=10000
            )
            
            # Position camera
            camera_pose = np.eye(4)
            camera_pose[2, 3] = 1000  # Move camera back
            self.scene.add(self.camera, pose=camera_pose)
            
            # Add lighting
            self._add_scene_lighting()
            
            print("3D renderer initialized successfully")
        except Exception as e:
            print(f"Error initializing renderer: {e}")
            self.renderer = None
            self.scene = None
    
    def _add_scene_lighting(self):
        """Add directional and point lights to the scene"""
        # Main directional light
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light1_pose = np.eye(4)
        light1_pose[:3, 3] = [0, 0, 10]
        self.scene.add(light1, pose=light1_pose)
        
        # Fill light
        light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [0, 10, 0]
        self.scene.add(light2, pose=light2_pose)
    
    def _get_person_mask(self, frame: np.ndarray) -> np.ndarray:
        """Fast person segmentation using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            return (results.segmentation_mask > 0.7).astype(np.uint8) * 255
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def _estimate_pose(self, frame: np.ndarray):
        """Estimate body pose using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)
    
    def _extract_shirt_region(self, frame: np.ndarray, pose_landmarks) -> Optional[np.ndarray]:
        """Create precise shirt region mask using pose landmarks"""
        if not pose_landmarks:
            return None
            
        h, w = frame.shape[:2]
        landmarks = pose_landmarks.landmark
        
        try:
            # Get key body landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            
            # Convert landmarks to pixel coordinates
            def to_px(lm): return (int(lm.x * w), int(lm.y * h))
            
            # Calculate body proportions
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
            torso_height = abs((left_shoulder.y + right_shoulder.y)/2 - 
                          (left_hip.y + right_hip.y)/2) * h
            
            # Create optimized shirt polygon
            nose_px = to_px(nose)
            neck_y = nose_px[1] + int(0.1 * h)
            
            left_shoulder_px = to_px(left_shoulder)
            right_shoulder_px = to_px(right_shoulder)
            shoulder_padding = int(shoulder_width * 0.1)
            
            left_elbow_px = to_px(left_elbow)
            right_elbow_px = to_px(right_elbow)
            left_wrist_px = to_px(left_wrist)
            right_wrist_px = to_px(right_wrist)
            
            left_hip_px = to_px(left_hip)
            right_hip_px = to_px(right_hip)
            shirt_bottom_y = int((left_hip_px[1] + right_hip_px[1]) / 2 - torso_height * 0.1)
            
            # Define shirt polygon points
            shirt_points = [
                # Collar area
                (nose_px[0] - int(shoulder_width * 0.15), neck_y),
                (nose_px[0] + int(shoulder_width * 0.15), neck_y),
                
                # Right side with sleeve curve
                (right_shoulder_px[0] + shoulder_padding, right_shoulder_px[1]),
                (right_elbow_px[0] + 20, right_shoulder_px[1] + 30),
                right_elbow_px,
                (right_wrist_px[0], right_wrist_px[1] - 15),
                
                # Torso right side
                (right_elbow_px[0] - 15, right_elbow_px[1] + 30),
                (right_shoulder_px[0], right_shoulder_px[1] + 40),
                (right_hip_px[0] - 15, shirt_bottom_y),
                
                # Bottom hem
                (left_hip_px[0] + 15, shirt_bottom_y),
                
                # Left side
                (left_shoulder_px[0], left_shoulder_px[1] + 40),
                (left_elbow_px[0] + 15, left_elbow_px[1] + 30),
                (left_wrist_px[0], left_wrist_px[1] - 15),
                left_elbow_px,
                (left_shoulder_px[0] - shoulder_padding, left_shoulder_px[1])
            ]
            
            # Create mask from polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(shirt_points, dtype=np.int32)], 255)
            return mask
            
        except (IndexError, AttributeError) as e:
            if self.debug_mode:
                print(f"Pose landmark error: {e}")
            return None
    
    def _refine_with_color(self, frame: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Refine shirt mask using color clustering"""
        if initial_mask is None:
            return None
            
        # Extract shirt region pixels
        shirt_pixels = frame[initial_mask == 255]
        if len(shirt_pixels) < 100:
            return initial_mask
            
        # Convert to LAB color space
        shirt_pixels_lab = cv2.cvtColor(shirt_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        shirt_pixels_lab = shirt_pixels_lab.reshape(-1, 3)
        
        # Adaptive clustering
        color_variance = np.var(shirt_pixels_lab, axis=0).mean()
        n_clusters = min(3 + int(color_variance / 20), 5)
        
        try:
            # Find dominant shirt color
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(shirt_pixels_lab)
            
            # Get dominant cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            # Adaptive tolerance
            cluster_std = np.std(shirt_pixels_lab[labels == dominant_cluster], axis=0).mean()
            self.color_tolerance = max(25, min(50, 30 + cluster_std * 0.5))
            
        except Exception as e:
            if self.debug_mode:
                print(f"Color clustering error: {e}")
            return initial_mask
        
        # Create color-based mask
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        color_diff = np.linalg.norm(frame_lab - dominant_color, axis=2)
        color_mask[color_diff < self.color_tolerance] = 255
        
        # Combine masks
        refined_mask = cv2.bitwise_and(initial_mask, color_mask)
        
        # Clean up with morphology
        kernel_size = max(3, min(7, int(frame.shape[1] / 150)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return refined_mask
    
    def _temporal_smoothing(self, current_mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to mask"""
        if current_mask is None:
            return None
            
        self.mask_history.append(current_mask.copy())
        
        if len(self.mask_history) < 3:
            return current_mask
        
        # Weighted temporal blending
        smoothed = np.zeros_like(current_mask, dtype=np.float32)
        weights = np.linspace(0.5, 1.0, len(self.mask_history))
        
        for i, mask in enumerate(self.mask_history):
            smoothed += mask.astype(np.float32) * weights[i]
        
        smoothed = (smoothed / weights.sum()).astype(np.uint8)
        _, smoothed = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return smoothed
    
    def _update_shirt_transform(self, pose_landmarks, frame_shape):
        """Update 3D shirt position based on pose"""
        if pose_landmarks is None or self.shirt_node is None:
            return
            
        h, w = frame_shape[:2]
        landmarks = pose_landmarks.landmark
        
        try:
            # Get key landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Calculate position
            center_x = (left_shoulder.x + right_shoulder.x) / 2 * w
            center_y = (left_shoulder.y + left_hip.y) / 2 * h
            
            # Calculate size
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
            torso_height = abs(left_shoulder.y - left_hip.y) * h
            shirt_size = max(shoulder_width, torso_height * 0.8)
            
            # Update with stabilization
            new_scale = shirt_size / 200.0
            self.shirt_scale = (self.stabilization_factor * self.shirt_scale + 
                               (1 - self.stabilization_factor) * new_scale)
            
            new_position = np.array([
                center_x - w/2,
                h/2 - center_y,
                0
            ])
            self.shirt_position = (
                self.stabilization_factor * self.shirt_position +
                (1 - self.stabilization_factor) * new_position
            )
            
            # Update rotation
            shoulder_angle = np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            )
            rotation_matrix = cv2.Rodrigues(np.array([0, 0, shoulder_angle]))[0]
            self.shirt_rotation = (
                self.stabilization_factor * self.shirt_rotation +
                (1 - self.stabilization_factor) * rotation_matrix
            )
            
            # Update scene
            shirt_pose = np.eye(4)
            shirt_pose[:3, :3] = self.shirt_rotation
            shirt_pose[:3, 3] = self.shirt_position
            shirt_pose[0, 0] = self.shirt_scale
            shirt_pose[1, 1] = self.shirt_scale
            shirt_pose[2, 2] = self.shirt_scale
            
            if self.shirt_node in self.scene.nodes:
                self.scene.set_pose(self.shirt_node, shirt_pose)
            else:
                self.scene.add(self.shirt_node, pose=shirt_pose)
                
        except (IndexError, AttributeError) as e:
            if self.debug_mode:
                print(f"Error updating shirt transform: {e}")
    
    def _render_3d_shirt(self, frame: np.ndarray) -> np.ndarray:
        """Render 3D shirt onto the frame"""
        if not self.show_3d or self.renderer is None:
            return frame
            
        try:
            # Render shirt with alpha
            color, _ = self.renderer.render(self.scene)
            color_bgra = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
            
            # Alpha blending
            alpha = color_bgra[:, :, 3] / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])
            shirt_rgb = color_bgra[:, :, :3]
            result = frame * (1 - alpha) + shirt_rgb * alpha
            
            return result.astype(np.uint8)
        except Exception as e:
            if self.debug_mode:
                print(f"Error rendering 3D shirt: {e}")
            return frame
    
    def _get_shirt_mask(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[mp.solutions.pose.PoseLandmark]]:
        """Main shirt segmentation pipeline"""
        self.frame_count += 1
        
        # Update FPS counter
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
        
        # Get person mask
        person_mask = self._get_person_mask(frame)
        
        # Skip pose detection on some frames for performance
        if self.frame_count % self.frame_skip == 0 or not self.pose_history:
            pose_results = self._estimate_pose(frame)
            if pose_results.pose_landmarks:
                self.pose_history.append(pose_results.pose_landmarks)
        
        # Use most recent pose
        current_pose = self.pose_history[-1] if self.pose_history else None
        
        if current_pose and person_mask is not None:
            # Extract shirt region
            shirt_roi = self._extract_shirt_region(frame, current_pose)
            
            if shirt_roi is not None:
                # Combine with person mask
                shirt_mask = cv2.bitwise_and(shirt_roi, person_mask)
                
                # Refine with color
                shirt_mask = self._refine_with_color(frame, shirt_mask)
                
                # Apply temporal smoothing
                shirt_mask = self._temporal_smoothing(shirt_mask)
                
                # Update 3D shirt
                self._update_shirt_transform(current_pose, frame.shape)
                
                return shirt_mask, current_pose
        
        return person_mask, current_pose
    
    def _visualize_results(self, frame: np.ndarray, shirt_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization outputs"""
        if shirt_mask is None:
            return frame, frame
            
        overlay = frame.copy()
        shirt_only = np.zeros_like(frame)
        
        # Apply shirt mask overlay
        if self.show_mask:
            shirt_overlay = np.zeros_like(frame)
            shirt_overlay[shirt_mask == 255] = self.shirt_color
            overlay = cv2.addWeighted(overlay, 0.7, shirt_overlay, 0.3, 0)
        
        # Create shirt-only view
        shirt_only[shirt_mask == 255] = frame[shirt_mask == 255]
        
        # Add 3D shirt
        if self.show_3d:
            overlay = self._render_3d_shirt(overlay)
            shirt_only = self._render_3d_shirt(shirt_only)
        
        # Add contours if enabled
        if self.show_outline:
            contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, self.outline_color, 2)
                cv2.drawContours(shirt_only, contours, -1, self.outline_color, 2)
        
        # Add fabric details
        if self.show_details:
            self._add_fabric_details(overlay, shirt_only, shirt_mask)
        
        return overlay, shirt_only
    
    def _add_fabric_details(self, overlay: np.ndarray, shirt_only: np.ndarray, mask: np.ndarray):
        """Add fabric texture and details to visualization"""
        # Edge detection
        gray = cv2.cvtColor(shirt_only, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[edges > 0] = self.detail_color
        shirt_only = cv2.addWeighted(shirt_only, 0.9, edges_colored, 0.1, 0)
        
        # Fabric texture pattern
        texture = np.zeros_like(shirt_only)
        
        # Vertical weave
        for i in range(0, shirt_only.shape[1], 4):
            cv2.line(texture, (i, 0), (i, shirt_only.shape[0]), (5, 5, 5), 1)
        
        # Horizontal weave
        for i in range(0, shirt_only.shape[0], 4):
            cv2.line(texture, (0, i), (shirt_only.shape[1], i), (5, 5, 5), 1)
        
        shirt_only = cv2.addWeighted(shirt_only, 0.9, texture, 0.05, 0)
    
    def _add_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """Add FPS counter to frame"""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single frame through the pipeline"""
        # Initialize renderer on first frame
        if not self.first_frame_processed and self.shirt_mesh:
            self._initialize_renderer(frame.shape[1], frame.shape[0])
            self.first_frame_processed = True
        
        # Resize for processing if needed
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        else:
            frame_resized = frame
            scale = 1.0
        
        # Mirror for natural interaction
        frame_resized = cv2.flip(frame_resized, 1)
        
        # Segment shirt
        shirt_mask, _ = self._get_shirt_mask(frame_resized)
        
        # Scale back if needed
        if scale != 1.0 and shirt_mask is not None:
            shirt_mask = cv2.resize(shirt_mask, (width, height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Create visualizations
        overlay, shirt_only = self._visualize_results(frame, shirt_mask)
        
        # Add FPS counter
        overlay = self._add_fps_counter(overlay)
        shirt_only = self._add_fps_counter(shirt_only)
        
        return overlay, shirt_only
    
    def run(self):
        """Main execution loop"""
        # Initialize camera
        cap = cv2.VideoCapture(self.CAMERA_ID, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        
        print("Virtual Try-On System Initialized")
        print("Controls:")
        print("  'm' - Toggle mask overlay")
        print("  'o' - Toggle outline display")
        print("  'd' - Toggle fabric details")
        print("  '3' - Toggle 3D shirt")
        print("  '1' - Overlay view")
        print("  '2' - Shirt-only view")
        print("  'b' - Toggle debug mode")
        print("  'q' - Quit")
        
        display_mode = 1  # 1=overlay, 2=shirt only
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Could not read frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                overlay, shirt_only = self.process_frame(frame)
                
                # Display based on mode
                if display_mode == 1:
                    cv2.imshow("Virtual Try-On (Overlay)", overlay)
                else:
                    cv2.imshow("Virtual Try-On (Shirt Only)", shirt_only)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_mask = not self.show_mask
                    print(f"Mask overlay: {'ON' if self.show_mask else 'OFF'}")
                elif key == ord('o'):
                    self.show_outline = not self.show_outline
                    print(f"Outline display: {'ON' if self.show_outline else 'OFF'}")
                elif key == ord('d'):
                    self.show_details = not self.show_details
                    print(f"Fabric details: {'ON' if self.show_details else 'OFF'}")
                elif key == ord('3'):
                    self.show_3d = not self.show_3d
                    print(f"3D shirt: {'ON' if self.show_3d else 'OFF'}")
                elif key == ord('1'):
                    display_mode = 1
                    cv2.destroyAllWindows()
                elif key == ord('2'):
                    display_mode = 2
                    cv2.destroyAllWindows()
                elif key == ord('b'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")

if __name__ == "__main__":
    try_on = VirtualTryOnSystem()
    try_on.run()
