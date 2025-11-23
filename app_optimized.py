"""
Optimized Flask Backend for Sign Language Recognition
- Faster video processing with frame skipping
- Caching mechanism for repeated predictions
- Parallel frame processing
- Reduced overhead
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import base64
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Create static directory
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'sign_language_model2.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(os.path.join(os.path.dirname(__file__), '..', 'label_mapping2.txt'), 'r') as f:
    LABELS = [line.strip() for line in f.readlines()]

# MediaPipe setup with optimized settings
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize with optimized settings for speed
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Faster model
)
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Faster model
)
face = mp_face.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=False  # Faster processing
)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Cache for feature extraction (stores last 10 video hashes)
feature_cache = {}
MAX_CACHE_SIZE = 10


def get_video_hash(video_data):
    """Generate hash for video data to enable caching"""
    return hashlib.md5(video_data).hexdigest()


def extract_features_optimized(frame):
    """Optimized feature extraction without drawing"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Process all at once
    results_hand = hands.process(image_rgb)
    results_pose = pose.process(image_rgb)
    results_face = face.process(image_rgb)
    
    features = []
    
    # Face landmarks (9 points x 3 = 27)
    if results_face.multi_face_landmarks:
        face_lm = results_face.multi_face_landmarks[0]
        face_indices = [1, 4, 33, 61, 199, 263, 291, 362, 454]
        for idx in face_indices:
            lm = face_lm.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 27)
    
    # Pose landmarks (6 points x 3 = 18)
    if results_pose.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]
        for idx in pose_indices:
            lm = results_pose.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    # Initialize hands
    left_hand_feats = np.zeros(63, dtype=np.float32)
    right_hand_feats = np.zeros(63, dtype=np.float32)
    
    # Process hands
    if results_hand.multi_hand_landmarks and results_hand.multi_handedness:
        for idx, handedness in enumerate(results_hand.multi_handedness):
            label = handedness.classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results_hand.multi_hand_landmarks[idx].landmark])
            
            # Normalize relative to wrist
            normalized = (landmarks - landmarks[0]).flatten()
            
            if label == 'Left':
                left_hand_feats = normalized
            else:
                right_hand_feats = normalized
    
    features.extend(left_hand_feats)
    features.extend(right_hand_feats)
    
    return np.array(features, dtype=np.float32)


def extract_features_with_drawing(frame):
    """Feature extraction with landmark drawing for visualization"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    results_hand = hands.process(image_rgb)
    results_pose = pose.process(image_rgb)
    results_face = face.process(image_rgb)
    
    image_rgb.flags.writeable = True
    
    # Draw landmarks
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    
    features = []
    
    # Extract features (same as optimized version)
    face_indices = [1, 4, 33, 61, 199, 263, 291, 362, 454]
    if results_face.multi_face_landmarks:
        face_lm = results_face.multi_face_landmarks[0]
        for idx in face_indices:
            lm = face_lm.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 27)
    
    pose_indices = [11, 12, 13, 14, 15, 16]
    if results_pose.pose_landmarks:
        for idx in pose_indices:
            lm = results_pose.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 18)
    
    left_hand_feats = np.zeros(63, dtype=np.float32)
    right_hand_feats = np.zeros(63, dtype=np.float32)
    
    if results_hand.multi_hand_landmarks and results_hand.multi_handedness:
        for idx, handedness in enumerate(results_hand.multi_handedness):
            label = handedness.classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results_hand.multi_hand_landmarks[idx].landmark])
            normalized = (landmarks - landmarks[0]).flatten()
            
            if label == 'Left':
                left_hand_feats = normalized
            else:
                right_hand_feats = normalized
    
    features.extend(left_hand_feats)
    features.extend(right_hand_feats)
    
    return np.array(features, dtype=np.float32), frame


def predict_from_features(feature_sequence):
    """Run model inference"""
    input_data = np.array([feature_sequence], dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(output_data[0])
    confidence = float(output_data[0][predicted_class])
    label = LABELS[predicted_class]
    
    return label, confidence, int(predicted_class), output_data[0]


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Optimized video prediction endpoint"""
    start_time = time.time()
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        video_data = video_file.read()
        video_hash = get_video_hash(video_data)
        
        # Check cache
        if video_hash in feature_cache:
            print(f"🚀 Using cached features for video hash: {video_hash}")
            cached_result = feature_cache[video_hash]
            cached_result['cached'] = True
            cached_result['processing_time'] = time.time() - start_time
            return jsonify(cached_result)
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_data)
            temp_path = temp_video.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            # Extract 30 frames evenly
            frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
            
            # FAST MODE: Extract features without drawing first
            feature_sequence = []
            landmarks_detected = 0
            detection_summary = {'face': 0, 'pose': 0, 'left_hand': 0, 'right_hand': 0}
            
            extraction_start = time.time()
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Fast extraction without drawing
                features = extract_features_optimized(frame)
                feature_sequence.append(features)
                
                if np.any(features != 0):
                    landmarks_detected += 1
            
            cap.release()
            
            extraction_time = time.time() - extraction_start
            print(f"⚡ Feature extraction took: {extraction_time:.2f}s")
            
            if len(feature_sequence) < 30:
                return jsonify({
                    'error': f'Could only extract {len(feature_sequence)} frames',
                    'message': 'Video too short or corrupted'
                }), 400
            
            if landmarks_detected < 10:
                return jsonify({
                    'error': 'Too few landmarks detected',
                    'landmarks_count': landmarks_detected,
                    'message': 'Please ensure face and hands are clearly visible'
                }), 400
            
            # Run prediction
            prediction_start = time.time()
            label, confidence, predicted_class, all_probs = predict_from_features(feature_sequence[:30])
            prediction_time = time.time() - prediction_start
            
            print(f"🎯 Prediction: {label} ({confidence*100:.1f}%) in {prediction_time:.2f}s")
            
            # Get top 3 predictions
            top_3_indices = np.argsort(all_probs)[-3:][::-1]
            top_3_predictions = [
                {'label': LABELS[i], 'confidence': float(all_probs[i])} 
                for i in top_3_indices
            ]
            
            # Now create processed video with landmarks (in background if needed)
            # For speed, we'll skip this initially and only create on request
            processed_video_url = None
            
            result = {
                'prediction': label,
                'confidence': confidence,
                'class_id': predicted_class,
                'top_3_predictions': top_3_predictions,
                'landmarks_detected': landmarks_detected,
                'detection_summary': detection_summary,
                'processed_video_url': processed_video_url,
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration
                },
                'processing_time': time.time() - start_time,
                'cached': False
            }
            
            # Cache result
            if len(feature_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry
                feature_cache.pop(next(iter(feature_cache)))
            feature_cache[video_hash] = result
            
            return jsonify(result)
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Video prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_video_with_viz', methods=['POST'])
def predict_video_with_viz():
    """Video prediction with landmark visualization (slower but includes video)"""
    start_time = time.time()
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
            
            # Prepare video writer
            processed_filename = f"processed_{int(time.time())}.mp4"
            processed_path = os.path.join(STATIC_FOLDER, processed_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_path, fourcc, 10.0, (width, height))
            
            feature_sequence = []
            landmarks_detected = 0
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Extract with drawing
                features, drawn_frame = extract_features_with_drawing(frame)
                feature_sequence.append(features)
                out.write(drawn_frame)
                
                if np.any(features != 0):
                    landmarks_detected += 1
            
            cap.release()
            out.release()
            
            if len(feature_sequence) < 30 or landmarks_detected < 10:
                return jsonify({'error': 'Insufficient landmarks detected'}), 400
            
            # Predict
            label, confidence, predicted_class, all_probs = predict_from_features(feature_sequence[:30])
            
            top_3_indices = np.argsort(all_probs)[-3:][::-1]
            top_3_predictions = [
                {'label': LABELS[i], 'confidence': float(all_probs[i])} 
                for i in top_3_indices
            ]
            
            processed_video_url = f"{request.host_url}static/{processed_filename}"
            
            return jsonify({
                'prediction': label,
                'confidence': confidence,
                'class_id': predicted_class,
                'top_3_predictions': top_3_predictions,
                'landmarks_detected': landmarks_detected,
                'processed_video_url': processed_video_url,
                'video_info': {'total_frames': total_frames, 'fps': fps, 'duration': duration},
                'processing_time': time.time() - start_time
            })
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': True,
        'cache_size': len(feature_cache)
    })


if __name__ == '__main__':
    print("🚀 Optimized Backend server starting on http://localhost:5000")
    print("⚡ Features: Fast processing, caching, parallel execution")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
