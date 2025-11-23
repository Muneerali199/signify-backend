import os
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import base64
import tempfile
import time

app = Flask(__name__)
CORS(app)

# Get port from environment variable for Render
PORT = int(os.environ.get('PORT', 5000))

# Create static directory for processed videos
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Load model (check both locations for compatibility)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sign_language_model2.tflite')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'sign_language_model2.tflite')

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels (check both locations)
LABEL_PATH = os.path.join(os.path.dirname(__file__), 'label_mapping2.txt')
if not os.path.exists(LABEL_PATH):
    LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'label_mapping2.txt')

with open(LABEL_PATH, 'r') as f:
    LABELS = [line.strip() for line in f.readlines()]

# MediaPipe setup - Matching predictionreal.py exactly
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize separate models to match training data distribution
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def extract_features_and_draw(frame):
    """Extract features and draw landmarks on frame using separate models"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Process with separate models
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
    detection_info = {
        'face': False,
        'pose': False,
        'left_hand': False,
        'right_hand': False
    }
    
    # Face landmarks (9 points x 3 = 27)
    if results_face.multi_face_landmarks:
        # Indices from predictionreal.py
        face_indices = [1, 4, 33, 61, 199, 263, 291, 362, 454]
        # Use first detected face
        face_lm = results_face.multi_face_landmarks[0]
        for idx in face_indices:
            lm = face_lm.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
        detection_info['face'] = True
    else:
        features.extend([0.0] * 27)
    
    # Pose landmarks (6 points x 3 = 18)
    if results_pose.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]
        for idx in pose_indices:
            lm = results_pose.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
        detection_info['pose'] = True
    else:
        features.extend([0.0] * 18)
    
    # Initialize hands as zeros
    left_hand_feats = np.zeros(63, dtype=np.float32)
    right_hand_feats = np.zeros(63, dtype=np.float32)
    
    # Process hands if detected
    if results_hand.multi_hand_landmarks and results_hand.multi_handedness:
        for idx, handedness in enumerate(results_hand.multi_handedness):
            label = handedness.classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results_hand.multi_hand_landmarks[idx].landmark])
            
            # Normalize relative to wrist
            normalized = (landmarks - landmarks[0]).flatten()
            
            if label == 'Left':
                left_hand_feats = normalized
                detection_info['left_hand'] = True
            else:
                right_hand_feats = normalized
                detection_info['right_hand'] = True
                
    features.extend(left_hand_feats)
    features.extend(right_hand_feats)
    
    return np.array(features, dtype=np.float32), detection_info, frame

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sign from video frames"""
    try:
        data = request.json
        frames_b64 = data.get('frames', [])
        
        print(f"📥 Received {len(frames_b64)} frames")
        
        if len(frames_b64) != 30:
            return jsonify({'error': f'Need exactly 30 frames, got {len(frames_b64)}'}), 400
        
        # Extract features from each frame
        feature_sequence = []
        landmarks_detected = 0
        detection_summary = {
            'face': 0,
            'pose': 0,
            'left_hand': 0,
            'right_hand': 0
        }
        
        for i, frame_b64 in enumerate(frames_b64):
            try:
                # Decode base64 image
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"⚠️ Frame {i} could not be decoded")
                    return jsonify({'error': f'Frame {i} decoding failed'}), 400
                
                print(f"Frame {i}: size={frame.shape}")
                
                # Extract features
                features, detection_info, _ = extract_features_and_draw(frame)
                feature_sequence.append(features)
                
                # Track detections
                if detection_info['face']:
                    detection_summary['face'] += 1
                if detection_info['pose']:
                    detection_summary['pose'] += 1
                if detection_info['left_hand']:
                    detection_summary['left_hand'] += 1
                if detection_info['right_hand']:
                    detection_summary['right_hand'] += 1
                
                # Check if landmarks were detected (non-zero features)
                if np.any(features != 0):
                    landmarks_detected += 1
                    
            except Exception as e:
                print(f"❌ Error processing frame {i}: {str(e)}")
                return jsonify({'error': f'Frame {i} processing failed: {str(e)}'}), 400
        
        print(f"✅ Landmarks detected in {landmarks_detected}/30 frames")
        print(f"📊 Detection summary: Face={detection_summary['face']}, Pose={detection_summary['pose']}, "
              f"LeftHand={detection_summary['left_hand']}, RightHand={detection_summary['right_hand']}")
        
        if landmarks_detected < 10:
            return jsonify({
                'error': 'Too few landmarks detected',
                'landmarks_count': landmarks_detected,
                'message': 'Please ensure face and hands are visible in camera'
            }), 400
        
        # Prepare input
        input_data = np.array([feature_sequence], dtype=np.float32)
        print(f"📊 Input shape: {input_data.shape}")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction
        predicted_class = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_class])
        label = LABELS[predicted_class]
        
        print(f"🎯 Prediction: {label} ({confidence*100:.1f}%)")
        
        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'class_id': int(predicted_class),
            'landmarks_detected': landmarks_detected,
            'detection_summary': detection_summary
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Predict sign from uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        print(f"📹 Received video file: {video_file.filename}")
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            
            # Extract 30 frames evenly
            frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
            
            feature_sequence = []
            landmarks_detected = 0
            detection_summary = {
                'face': 0,
                'pose': 0,
                'left_hand': 0,
                'right_hand': 0
            }
            
            # Prepare video writer for processed video
            processed_filename = f"processed_{int(time.time())}.mp4"
            processed_path = os.path.join(STATIC_FOLDER, processed_filename)
            
            # Get frame size from first frame
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_path, fourcc, 10.0, (width, height))
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"⚠️ Could not read frame {frame_idx}")
                    continue
                
                # Extract features and draw landmarks
                features, detection_info, drawn_frame = extract_features_and_draw(frame)
                feature_sequence.append(features)
                
                # Write frame with landmarks to output video
                out.write(drawn_frame)
                
                # Track detections
                if detection_info['face']:
                    detection_summary['face'] += 1
                if detection_info['pose']:
                    detection_summary['pose'] += 1
                if detection_info['left_hand']:
                    detection_summary['left_hand'] += 1
                if detection_info['right_hand']:
                    detection_summary['right_hand'] += 1
                
                if np.any(features != 0):
                    landmarks_detected += 1
            
            cap.release()
            out.release()
            
            processed_video_url = f"{request.host_url}static/{processed_filename}"
            print(f"✅ Processed video saved to: {processed_path}")
            
            print(f"✅ Processed {len(feature_sequence)} frames")
            print(f"✅ Landmarks detected in {landmarks_detected}/30 frames")
            print(f"📊 Detection summary: Face={detection_summary['face']}, Pose={detection_summary['pose']}, "
                  f"LeftHand={detection_summary['left_hand']}, RightHand={detection_summary['right_hand']}")
            
            if len(feature_sequence) < 30:
                return jsonify({
                    'error': f'Could only extract {len(feature_sequence)} frames',
                    'message': 'Video too short or corrupted'
                }), 400
            
            if landmarks_detected < 10:
                return jsonify({
                    'error': 'Too few landmarks detected',
                    'landmarks_count': landmarks_detected,
                    'detection_summary': detection_summary,
                    'message': 'Please ensure face and hands are clearly visible'
                }), 400
            
            # Prepare input
            input_data = np.array([feature_sequence[:30]], dtype=np.float32)
            print(f"📊 Input shape: {input_data.shape}")
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Get prediction
            predicted_class = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_class])
            label = LABELS[predicted_class]
            
            # Get top 3 predictions for better debugging
            top_3_indices = np.argsort(output_data[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'label': LABELS[i].split(',')[0] if ',' in LABELS[i] else LABELS[i],
                    'confidence': float(output_data[0][i]),
                    'class_id': int(i)
                }
                for i in top_3_indices
            ]
            
            print(f"🎯 Prediction: {label} ({confidence*100:.1f}%)")
            print(f"📊 Top 3: {top_3_predictions}")
            
            # Extract label name without ID
            label_name = label.split(',')[0] if ',' in label else label
            
            # Confidence threshold check
            MIN_CONFIDENCE = 0.65
            if confidence < MIN_CONFIDENCE:
                print(f"⚠️ Low confidence: {confidence:.2f} < {MIN_CONFIDENCE}")
                return jsonify({
                    'error': 'Low confidence prediction',
                    'message': 'Please perform the sign more clearly and ensure good lighting',
                    'confidence': confidence,
                    'top_predictions': top_3_predictions,
                    'landmarks_detected': landmarks_detected
                }), 400
            
            return jsonify({
                'prediction': label_name,
                'confidence': confidence,
                'class_id': int(predicted_class),
                'top_3_predictions': top_3_predictions,
                'landmarks_detected': landmarks_detected,
                'detection_summary': detection_summary,
                'processed_video_url': processed_video_url,
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration
                }
            })
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Video prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    print(f"🚀 Backend server starting on http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=True)
