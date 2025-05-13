from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime
import os
import json
from flask_socketio import SocketIO
import logging
import threading
import queue
import itertools
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'atomvision2025'
socketio = SocketIO(app, async_mode='threading')

# Global variables
active_detectors = {}
detection_settings = {}
log_folder = "logs"
frame_queue = queue.Queue(maxsize=20)
frame_skip_counter = 0
detector_cycle = None
yolo_model = None

# Ensure directories exist
os.makedirs(log_folder, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize YOLO model
def load_yolo_model():
    global yolo_model
    model_path = "best.pt"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        yolo_model = YOLO(model_path)
        yolo_model.to('cpu')  # Ensure CPU inference
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        return False

# Feature to model mapping
feature_to_model = {
    'mask-detection': {'class_id': 1, 'function': 'detect_yolo'},
    'weapon-like-object': {'class_id': 2, 'function': 'detect_yolo'},
    'fire-smoke-detection': {'class_id': 0, 'function': 'detect_yolo'},
    # Other features (not supported by YOLO model, using simulation)
    'uniform-check': {'function': 'detect_simulation'},
    'apron-detection': {'function': 'detect_simulation'},
    'id-badge-visibility': {'function': 'detect_simulation'},
    'shoe-cover-check': {'function': 'detect_simulation'},
    'loitering-detection': {'function': 'detect_simulation'},
    'unusual-movement': {'function': 'detect_simulation'},
    'object-left-behind': {'function': 'detect_simulation'},
    'face-not-recognized': {'function': 'detect_simulation'},
    'perimeter-breach-detection': {'function': 'detect_simulation'},
    'crowd-density-zone': {'function': 'detect_simulation'},
    'queue-detection': {'function': 'detect_simulation'},
    'aggressive-posture': {'function': 'detect_simulation'},
    'falling-detection': {'function': 'detect_simulation'},
    'lights-off-detection': {'function': 'detect_simulation'},
    'door-open-close-monitor': {'function': 'detect_simulation'},
    'explosion-detection': {'function': 'detect_simulation'},
    'commotion-detection': {'function': 'detect_simulation'},
    'inappropriate-behavior-detection': {'function': 'detect_simulation'},
    'panic-behavior-detection': {'function': 'detect_simulation'}
}

def decode_frame(base64_string):
    try:
        if not base64_string.startswith('data:image'):
            raise ValueError("Invalid base64 string format")
        img_data = base64.b64decode(base64_string.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {str(e)}")
        return None

def detect_yolo(frame, settings=None, feature=None):
    if frame is None or yolo_model is None:
        return frame, []
    
    detections = []
    try:
        # Run YOLO inference
        results = yolo_model.predict(source=frame, conf=0.25, iou=0.45, save=False, verbose=False)
        class_id = feature_to_model[feature]['class_id']
        
        for result in results:
            for box in result.boxes:
                if int(box.cls) == class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detection_info = {
                        'type': feature,
                        'confidence': float(box.conf),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'bbox': [x1, y1, x2, y2]
                    }
                    if feature == 'mask-detection':
                        detection_info['mask_detected'] = True
                    elif feature == 'weapon-like-object':
                        detection_info['weapon_detected'] = True
                    elif feature == 'fire-smoke-detection':
                        detection_info['fire_smoke_detected'] = True
                    detections.append(detection_info)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{feature.replace('-', ' ')}: {box.conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logger.debug(f"YOLO detections for {feature}: {len(detections)}")
    except Exception as e:
        logger.error(f"Error in YOLO detection for {feature}: {str(e)}\n{traceback.format_exc()}")

    return frame, detections

def detect_simulation(frame, settings=None, feature=None):
    if frame is None:
        return frame, []
    
    detections = []
    if np.random.random() < 0.3:
        detection_info = {
            'type': feature,
            'confidence': round(np.random.random() * 0.5 + 0.5, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bbox': [100, 100, 200, 200]
        }
        detections.append(detection_info)
        logger.debug(f"Simulation detection for {feature}")
    
    return frame, detections

# Map detector functions
detection_functions = {
    'detect_yolo': detect_yolo,
    'detect_simulation': detect_simulation
}

def process_frame_worker():
    global detector_cycle
    log_buffer = []
    while True:
        try:
            base64_frame = frame_queue.get(timeout=1.0)
            frame = decode_frame(base64_frame)
            if frame is None:
                frame_queue.task_done()
                continue
            
            global frame_skip_counter
            frame_skip_counter += 1
            if frame_skip_counter % 4 != 0:
                frame_queue.task_done()
                continue
            
            logger.debug(f"Current frame queue size: {frame_queue.qsize()}")
            
            processed_frame = frame.copy()
            detections = []
            
            active_features = [f for f, active in active_detectors.items() if active and f in feature_to_model]
            if active_features and not detector_cycle:
                logger.debug(f"Active detectors: {active_features}")
                detector_cycle = itertools.cycle(active_features)
            
            if active_features:
                feature = next(detector_cycle)
                model_info = feature_to_model[feature]
                func_name = model_info['function']
                
                logger.debug(f"Processing detection for: {feature}")
                
                if func_name in detection_functions:
                    settings = detection_settings.get(feature, {})
                    processed_frame, feature_detections = detection_functions[func_name](processed_frame, settings, feature)
                    detections.extend(feature_detections)
            
            for detection in detections:
                # Check for positive detections
                is_positive = (
                    (detection['type'] == 'mask-detection' and detection.get('mask_detected', False)) or
                    (detection['type'] == 'weapon-like-object' and detection.get('weapon_detected', False)) or
                    (detection['type'] == 'fire-smoke-detection' and detection.get('fire_smoke_detected', False)) or
                    (detection['type'] not in ['mask-detection', 'weapon-like-object', 'fire-smoke-detection'])
                )
                
                if not is_positive:
                    continue
                
                timestamp = datetime.now()
                log_entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "feature": detection['type'],
                    "confidence": detection.get('confidence', 0.0),
                    "mask_detected": str(detection.get('mask_detected', None)).lower(),
                    "weapon_detected": str(detection.get('weapon_detected', None)).lower(),
                    "fire_smoke_detected": str(detection.get('fire_smoke_detected', None)).lower()
                }
                log_buffer.append(log_entry)
                
                if detection['type'] == 'mask-detection':
                    status = 'Mask'
                elif detection['type'] == 'weapon-like-object':
                    status = 'Weapon'
                elif detection['type'] == 'fire-smoke-detection':
                    status = 'Fire/Smoke'
                else:
                    status = 'Detected'
                
                socketio.emit('detection_alert', {
                    'message': f"Alert: {detection['type'].replace('-', ' ')} detected ({status})",
                    'feature': detection['type'],
                    'timestamp': timestamp.strftime("%H:%M:%S"),
                    'confidence': detection.get('confidence', 0.0)
                })
            
            if len(log_buffer) >= 10:
                try:
                    with open(f"{log_folder}/detection_log.json", "a") as log_file:
                        for entry in log_buffer:
                            log_file.write(json.dumps(entry) + "\n")
                    log_buffer.clear()
                except Exception as e:
                    logger.error(f"Error writing logs: {str(e)}")
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('processed_frame', {'frame': f'data:image/jpeg;base64,{frame_b64}'})
            
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}\n{traceback.format_exc()}")
            frame_queue.task_done()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/toggle_feature', methods=['POST'])
def toggle_feature():
    try:
        data = request.json
        feature = data.get('feature')
        active = data.get('active', False)
        
        logger.debug(f"Received toggle request for feature: {feature}")
        
        if feature in feature_to_model:
            active_detectors[feature] = active
            logger.info(f"Feature {feature} set to {active}")
            socketio.emit('status_update', {'feature': feature, 'active': active})
            global detector_cycle
            detector_cycle = None
            return jsonify({"success": True})
        
        logger.error(f"Invalid feature specified: {feature}. Available features: {list(feature_to_model.keys())}")
        return jsonify({"success": False, "error": f"Invalid feature specified: {feature}"})
    except Exception as e:
        logger.error(f"Error toggling feature: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        feature = data.get('feature')
        settings = data.get('settings', {})
        
        if feature in feature_to_model:
            detection_settings[feature] = settings
            logger.info(f"Updated settings for {feature}")
            return jsonify({"success": True})
        
        logger.error(f"Invalid feature for settings update: {feature}")
        return jsonify({"success": False, "error": "Invalid feature specified"})
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/system_settings', methods=['POST'])
def update_system_settings():
    try:
        global log_folder
        data = request.json
        
        if 'log_folder' in data:
            log_folder = data['log_folder']
            os.makedirs(log_folder, exist_ok=True)
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating system settings: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    socketio.emit('status_update', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('send_frame')
def receive_frame(data):
    try:
        if 'frame' in data:
            if frame_queue.full():
                logger.warning("Frame queue full, dropping frame")
                return
            frame_queue.put(data['frame'])
    except Exception as e:
        logger.error(f"Error queuing frame: {str(e)}")

if __name__ == '__main__':
    if load_yolo_model():
        threading.Thread(target=process_frame_worker, daemon=True).start()
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to start server due to model loading failure")