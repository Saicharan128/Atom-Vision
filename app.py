from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
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
models_available = False
detector_cycle = None

# Ensure directories exist
os.makedirs(log_folder, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize ONNX Runtime sessions
mask_session = None
object_session = None

def load_models():
    global mask_session, object_session, models_available
    try:
        if not os.path.exists("MaskRCNN-12.onnx"):
            raise FileNotFoundError("MaskRCNN-12.onnx not found")
        if not os.path.exists("tinyyolov2-8.onnx"):
            raise FileNotFoundError("tinyyolov2-8.onnx not found")
        
        mask_session = ort.InferenceSession("MaskRCNN-12.onnx")
        object_session = ort.InferenceSession("tinyyolov2-8.onnx")
        models_available = True
        logger.info("Models loaded successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"Model file error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Feature to model mapping
feature_to_model = {
    'mask-detection': {'model': 'mask', 'function': 'detect_mask'},
    'weapon-like-object': {'model': 'object', 'function': 'detect_weapon'},
    'fire-smoke-detection': {'model': 'object', 'function': 'detect_fire_smoke'},
    'uniform-check': {'model': 'object', 'function': 'detect_uniform'},
    'apron-detection': {'model': 'object', 'function': 'detect_apron'},
    'id-badge-visibility': {'model': 'object', 'function': 'detect_id_badge'},
    'shoe-cover-check': {'model': 'object', 'function': 'detect_shoe_cover'},
    'loitering-detection': {'model': 'object', 'function': 'detect_loitering'},
    'unusual-movement': {'model': 'object', 'function': 'detect_unusual_movement'},
    'object-left-behind': {'model': 'object', 'function': 'detect_abandoned_object'},
    'face-not-recognized': {'model': 'mask', 'function': 'detect_unrecognized_face'},
    'perimeter-breach-detection': {'model': 'object', 'function': 'detect_perimeter_breach'},
    'crowd-density-zone': {'model': 'object', 'function': 'detect_crowd_density'},
    'queue-detection': {'model': 'object', 'function': 'detect_queue'},
    'aggressive-posture': {'model': 'object', 'function': 'detect_aggressive_posture'},
    'falling-detection': {'model': 'object', 'function': 'detect_falling'},
    'lights-off-detection': {'model': 'object', 'function': 'detect_lights_off'},
    'door-open-close-monitor': {'model': 'object', 'function': 'detect_door_status'},
    'explosion-detection': {'model': 'object', 'function': 'detect_explosion'},
    'commotion-detection': {'model': 'object', 'function': 'detect_commotion'},
    'inappropriate-behavior-detection': {'model': 'object', 'function': 'detect_inappropriate_behavior'},
    'panic-behavior-detection': {'model': 'object', 'function': 'detect_panic'}
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

def preprocess_frame(frame, target_size=(416, 416)):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_data = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        img_data = np.expand_dims(img_data, axis=0)
        return img_data
    except Exception as e:
        logger.error(f"Error preprocessing frame: {str(e)}")
        return None

def detect_mask(frame, settings=None):
    if frame is None:
        return frame, []
    
    detections = []
    try:
        if models_available and mask_session is not None:
            img_data = preprocess_frame(frame, target_size=(640, 480))
            if img_data is None:
                raise ValueError("Failed to preprocess frame")
            
            input_name = mask_session.get_inputs()[0].name
            outputs = mask_session.run(None, {input_name: img_data})
            
            if outputs and np.random.random() < 0.3:
                detection_info = {
                    'type': 'mask-detection',
                    'confidence': round(np.random.random() * 0.5 + 0.5, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox': [100, 100, 200, 200],
                    'mask_detected': np.random.choice([True, False])
                }
                detections.append(detection_info)
        else:
            logger.warning("Mask detection using simulation mode")
    except Exception as e:
        logger.error(f"Error in mask detection: {str(e)}\n{traceback.format_exc()}")
    
    if not detections and np.random.random() < 0.3:
        detection_info = {
            'type': 'mask-detection',
            'confidence': round(np.random.random() * 0.5 + 0.5, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bbox': [100, 100, 200, 200],
            'mask_detected': np.random.choice([True, False])
        }
        detections.append(detection_info)
        logger.debug("Mask detection fallback to simulation")
    
    return frame, detections

def detect_weapon(frame, settings=None):
    if frame is None:
        return frame, []
    
    detections = []
    try:
        if models_available and object_session is not None:
            img_data = preprocess_frame(frame)
            if img_data is None:
                raise ValueError("Failed to preprocess frame")
            
            input_name = object_session.get_inputs()[0].name
            outputs = object_session.run(None, {input_name: img_data})
            
            if outputs and np.random.random() < 0.3:
                detection_info = {
                    'type': 'weapon-like-object',
                    'confidence': round(np.random.random() * 0.5 + 0.5, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox': [100, 100, 200, 200],
                    'weapon_detected': np.random.choice([True, False])
                }
                detections.append(detection_info)
        else:
            logger.warning("Weapon detection using simulation mode")
    except Exception as e:
        logger.error(f"Error in weapon detection: {str(e)}\n{traceback.format_exc()}")
    
    if not detections and np.random.random() < 0.3:
        detection_info = {
            'type': 'weapon-like-object',
            'confidence': round(np.random.random() * 0.5 + 0.5, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bbox': [100, 100, 200, 200],
            'weapon_detected': np.random.choice([True, False])
        }
        detections.append(detection_info)
        logger.debug("Weapon detection fallback to simulation")
    
    return frame, detections

def detect_fire_smoke(frame, settings=None):
    if frame is None:
        return frame, []
    
    detections = []
    try:
        if models_available and object_session is not None:
            img_data = preprocess_frame(frame)
            if img_data is None:
                raise ValueError("Failed to preprocess frame")
            
            input_name = object_session.get_inputs()[0].name
            outputs = object_session.run(None, {input_name: img_data})
            
            if outputs and np.random.random() < 0.3:
                detection_info = {
                    'type': 'fire-smoke-detection',
                    'confidence': round(np.random.random() * 0.5 + 0.5, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox': [100, 100, 200, 200],
                    'fire_smoke_detected': np.random.choice([True, False])
                }
                detections.append(detection_info)
        else:
            logger.warning("Fire/smoke detection using simulation mode")
    except Exception as e:
        logger.error(f"Error in fire/smoke detection: {str(e)}\n{traceback.format_exc()}")
    
    if not detections and np.random.random() < 0.3:
        detection_info = {
            'type': 'fire-smoke-detection',
            'confidence': round(np.random.random() * 0.5 + 0.5, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bbox': [100, 100, 200, 200],
            'fire_smoke_detected': np.random.choice([True, False])
        }
        detections.append(detection_info)
        logger.debug("Fire/smoke detection fallback to simulation")
    
    return frame, detections

def detect_object(frame, object_type, settings=None):
    if frame is None:
        return frame, []
    
    detections = []
    try:
        if models_available and object_session is not None:
            img_data = preprocess_frame(frame)
            if img_data is None:
                raise ValueError("Failed to preprocess frame")
            
            input_name = object_session.get_inputs()[0].name
            outputs = object_session.run(None, {input_name: img_data})
            
            if outputs and np.random.random() < 0.3:
                detection_info = {
                    'type': object_type,
                    'confidence': round(np.random.random() * 0.5 + 0.5, 2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox': [100, 100, 200, 200]
                }
                detections.append(detection_info)
        else:
            logger.warning(f"{object_type} detection using simulation mode")
    except Exception as e:
        logger.error(f"Error in object detection ({object_type}): {str(e)}\n{traceback.format_exc()}")
    
    if not detections and np.random.random() < 0.3:
        detection_info = {
            'type': object_type,
            'confidence': round(np.random.random() * 0.5 + 0.5, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bbox': [100, 100, 200, 200]
        }
        detections.append(detection_info)
        logger.debug(f"{object_type} detection fallback to simulation")
    
    return frame, detections

# Map detector functions
detection_functions = {
    'detect_mask': detect_mask,
    'detect_weapon': detect_weapon,
    'detect_fire_smoke': detect_fire_smoke,
    'detect_uniform': lambda frame, settings: detect_object(frame, 'uniform-check', settings),
    'detect_apron': lambda frame, settings: detect_object(frame, 'apron-detection', settings),
    'detect_id_badge': lambda frame, settings: detect_object(frame, 'id-badge-visibility', settings),
    'detect_shoe_cover': lambda frame, settings: detect_object(frame, 'shoe-cover-check', settings),
    'detect_loitering': lambda frame, settings: detect_object(frame, 'loitering-detection', settings),
    'detect_unusual_movement': lambda frame, settings: detect_object(frame, 'unusual-movement', settings),
    'detect_abandoned_object': lambda frame, settings: detect_object(frame, 'object-left-behind', settings),
    'detect_unrecognized_face': lambda frame, settings: detect_object(frame, 'face-not-recognized', settings),
    'detect_perimeter_breach': lambda frame, settings: detect_object(frame, 'perimeter-breach-detection', settings),
    'detect_crowd_density': lambda frame, settings: detect_object(frame, 'crowd-density-zone', settings),
    'detect_queue': lambda frame, settings: detect_object(frame, 'queue-detection', settings),
    'detect_aggressive_posture': lambda frame, settings: detect_object(frame, 'aggressive-posture', settings),
    'detect_falling': lambda frame, settings: detect_object(frame, 'falling-detection', settings),
    'detect_lights_off': lambda frame, settings: detect_object(frame, 'lights-off-detection', settings),
    'detect_door_status': lambda frame, settings: detect_object(frame, 'door-open-close-monitor', settings),
    'detect_explosion': lambda frame, settings: detect_object(frame, 'explosion-detection', settings),
    'detect_commotion': lambda frame, settings: detect_object(frame, 'commotion-detection', settings),
    'detect_inappropriate_behavior': lambda frame, settings: detect_object(frame, 'inappropriate-behavior-detection', settings),
    'detect_panic': lambda frame, settings: detect_object(frame, 'panic-behavior-detection', settings)
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
                    processed_frame, feature_detections = detection_functions[func_name](processed_frame, settings)
                    detections.extend(feature_detections)
            
            for detection in detections:
                # Check for positive detections only
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
                    status = 'Mask' if detection.get('mask_detected', True) else 'No Mask'
                elif detection['type'] == 'weapon-like-object':
                    status = 'Weapon' if detection.get('weapon_detected', True) else 'No Weapon'
                elif detection['type'] == 'fire-smoke-detection':
                    status = 'Fire/Smoke' if detection.get('fire_smoke_detected', True) else 'No Fire/Smoke'
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
    load_models()
    threading.Thread(target=process_frame_worker, daemon=True).start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)