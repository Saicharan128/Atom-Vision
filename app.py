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
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'atomvision2025'
socketio = SocketIO(app, async_mode='threading')

# Global variables
active_detectors = {
    'fire-smoke-detection': False,
    'mask-detection': False,
    'weapon-like-object': False
}
detection_settings = {}
log_folder = "logs"
log_file_name = "detection_log.json"
frame_queue = queue.Queue(maxsize=20)
frame_skip_counter = 0
detector_cycle = None
yolo_model = None
cnn_model = None
face_cascade = None

# Ensure directories exist
os.makedirs(log_folder, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Define CNN model (must match the training script)
class MaskNoMaskCNN(nn.Module):
    def __init__(self):
        super(MaskNoMaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: with_mask, without_mask
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize YOLO, CNN models, and face cascade
def load_models():
    global yolo_model, cnn_model, face_cascade
    # Load YOLO model for fire/smoke and weapon detection
    model_path = "best.pt"
    try:
        if not os.path.exists(model_path):
            logger.warning(f"YOLO model not found at {model_path}. Fire/smoke and weapon detection will be disabled.")
        else:
            yolo_model = YOLO(model_path)
            yolo_model.to('cpu')
            logger.info("YOLO model loaded successfully from %s", model_path)
    except Exception as e:
        logger.error("Error loading YOLO model: %s\n%s", str(e), traceback.format_exc())

    # Load CNN model for mask detection
    cnn_model_path = "C:/Users/saich/Downloads/New/withmask_withoutmask_model.pt"
    try:
        if not os.path.exists(cnn_model_path):
            raise FileNotFoundError(f"CNN model not found at {cnn_model_path}.")
        cnn_model = MaskNoMaskCNN()
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location='cpu'))
        cnn_model.to('cpu')
        cnn_model.eval()
        logger.info("CNN model loaded successfully from %s", cnn_model_path)
    except Exception as e:
        logger.error("Error loading CNN model: %s\n%s", str(e), traceback.format_exc())
        return False

    # Load face cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise FileNotFoundError("Failed to load Haar cascade file.")
        logger.info("Face cascade loaded successfully")
    except Exception as e:
        logger.error("Error loading face cascade: %s\n%s", str(e), traceback.format_exc())
        face_cascade = None

    return True

# Define transforms for CNN model (same as training)
cnn_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Decode frame from base64
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
        logger.error("Error decoding frame: %s", str(e))
        return None

# YOLO detection function (for fire/smoke and weapon)
def detect_yolo(frame, settings=None, feature=None):
    if frame is None or yolo_model is None:
        return frame, []
    
    detections = []
    try:
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
                        'bbox': [x1, y1, x2, y2],
                        'display_name': feature_to_model[feature]['display_name']
                    }
                    if feature == 'weapon-like-object':
                        detection_info['weapon_detected'] = True
                    elif feature == 'fire-smoke-detection':
                        detection_info['fire_smoke_detected'] = True
                    detections.append(detection_info)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{feature_to_model[feature]['display_name']}: {box.conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logger.debug("YOLO detections for %s: %d", feature, len(detections))
    except Exception as e:
        logger.error("Error in YOLO detection for %s: %s\n%s", feature, str(e), traceback.format_exc())

    return frame, detections

# CNN detection function for mask detection with face detection
def detect_cnn(frame, settings=None, feature=None):
    if frame is None or cnn_model is None:
        return frame, []
    
    detections = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)) if face_cascade else []
        
        if len(faces) == 0:
            logger.debug("No faces detected in frame")
            detection_info = {
                'type': 'mask-detection',
                'confidence': 0.0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'bbox': None,
                'display_name': 'No Face',
                'mask_detected': False
            }
            detections.append(detection_info)
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame, detections

        for (x, y, w, h) in faces:
            # Tighten face crop to reduce background
            padding = int(min(w, h) * 0.1)
            x1 = max(x + padding, 0)
            y1 = max(y + padding, 0)
            x2 = min(x + w - padding, frame.shape[1])
            y2 = min(y + h - padding, frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            face = frame[y1:y2, x1:x2]
            
            frame_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            input_tensor = cnn_transforms(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                input_tensor = input_tensor.to('cpu')
                outputs = cnn_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            class_names = ['With Mask', 'Without Mask']
            prediction = class_names[predicted.item()]
            confidence_score = confidence.item()
            
            logger.debug("Face crop: x=%d, y=%d, w=%d, h=%d", x1, y1, x2-x1, y2-y1)
            logger.debug("CNN raw outputs: %s, probabilities: %s, prediction: %s (confidence: %.2f)",
                        outputs.tolist(), probabilities.tolist(), prediction, confidence_score)
            
            # Apply confidence threshold
            if confidence_score < 0.7:
                logger.debug("Confidence below threshold (0.7), skipping detection")
                continue
                
            detection_info = {
                'type': 'mask-detection',
                'confidence': confidence_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'bbox': [x1, y1, x2, y2],
                'display_name': 'Mask' if prediction == 'With Mask' else 'No Mask',
                'mask_detected': prediction == 'With Mask'
            }
            detections.append(detection_info)
            
            color = (0, 255, 0) if prediction == 'With Mask' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection_info['display_name']}: {confidence_score:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    except Exception as e:
        logger.error("Error in CNN detection: %s\n%s", str(e), traceback.format_exc())

    return frame, detections

# Map detector functions
detection_functions = {
    'detect_yolo': detect_yolo,
    'detect_cnn': detect_cnn
}

# Feature to model mapping
feature_to_model = {
    'fire-smoke-detection': {'class_id': 0, 'function': 'detect_yolo', 'display_name': 'Fire/Smoke'},
    'mask-detection': {'class_id': None, 'function': 'detect_cnn', 'display_name': 'Mask'},
    'weapon-like-object': {'class_id': 2, 'function': 'detect_yolo', 'display_name': 'Weapon'}
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
            
            logger.debug("Current frame queue size: %d", frame_queue.qsize())
            
            processed_frame = frame.copy()
            detections = []
            
            active_features = [f for f, active in active_detectors.items() if active and f in feature_to_model]
            if not active_features:
                logger.debug("No active detectors, skipping frame processing")
                _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('processed_frame', {'frame': f'data:image/jpeg;base64,{frame_b64}'})
                frame_queue.task_done()
                continue
            
            if active_features and not detector_cycle:
                logger.debug("Active detectors: %s", active_features)
                detector_cycle = itertools.cycle(active_features)
            
            feature = next(detector_cycle)
            model_info = feature_to_model[feature]
            func_name = model_info['function']
            
            logger.debug("Processing detection for: %s", feature)
            
            no_mask_detected = False
            settings = detection_settings.get(feature, {})
            processed_frame, feature_detections = detection_functions[func_name](processed_frame, settings, feature)
            detections.extend(feature_detections)
            if feature == 'mask-detection' and not any(d.get('mask_detected', False) for d in feature_detections):
                no_mask_detected = True
            
            for detection in detections:
                is_positive = (
                    (detection['type'] == 'mask-detection' and detection.get('mask_detected', False)) or
                    (detection['type'] == 'weapon-like-object' and detection.get('weapon_detected', False)) or
                    (detection['type'] == 'fire-smoke-detection' and detection.get('fire_smoke_detected', False))
                )
                
                if not is_positive:
                    continue
                
                timestamp = datetime.now()
                log_entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "feature": detection['type'],
                    "confidence": detection.get('confidence', 0.0),
                    "mask_detected": str(detection.get('mask_detected', False)).lower(),
                    "weapon_detected": str(detection.get('weapon_detected', False)).lower(),
                    "fire_smoke_detected": str(detection.get('fire_smoke_detected', False)).lower(),
                    "no_mask_detected": "false"
                }
                log_buffer.append(log_entry)
                logger.debug("Logged detection for %s with confidence %s", detection['type'], detection.get('confidence', 0.0))
                
                socketio.emit('detection_alert', {
                    'message': f"Alert: {detection['display_name']} detected",
                    'feature': detection['type'],
                    'timestamp': timestamp.strftime("%H:%M:%S"),
                    'confidence': detection.get('confidence', 0.0)
                })
            
            if no_mask_detected and active_detectors['mask-detection']:
                timestamp = datetime.now()
                logger.debug("No mask detected for mask-detection, emitting UI alert")
                socketio.emit('detection_alert', {
                    'message': "Alert: No Mask detected",
                    'feature': 'no-mask-detection',
                    'timestamp': timestamp.strftime("%H:%M:%S"),
                    'confidence': 0.0
                })
            
            if len(log_buffer) >= 10:
                try:
                    log_path = os.path.join(log_folder, log_file_name)
                    with open(log_path, "a") as log_file:
                        for entry in log_buffer:
                            log_file.write(json.dumps(entry) + "\n")
                    log_buffer.clear()
                    logger.debug("Wrote %d log entries to %s", 10, log_path)
                except Exception as e:
                    logger.error("Error writing logs: %s", str(e))
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('processed_frame', {'frame': f'data:image/jpeg;base64,{frame_b64}'})
            
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Error processing frame: %s\n%s", str(e), traceback.format_exc())
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
        
        logger.info("Toggle request for feature: %s, active: %s", feature, active)
        
        if feature in feature_to_model:
            active_detectors[feature] = active
            logger.info("Feature %s set to %s", feature, active)
            socketio.emit('status_update', {'feature': feature, 'active': active})
            global detector_cycle
            detector_cycle = None
            return jsonify({"success": True})
        
        logger.warning("Unsupported feature toggled: %s. Available features: %s", feature, list(feature_to_model.keys()))
        return jsonify({"success": False, "error": f"Unsupported feature: {feature}. Contact support to enable."})
    except Exception as e:
        logger.error("Error toggling feature: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        feature = data.get('feature')
        settings = data.get('settings', {})
        
        if feature in feature_to_model:
            detection_settings[feature] = settings
            logger.info("Updated settings for %s: %s", feature, settings)
            return jsonify({"success": True})
        
        logger.warning("Unsupported feature for settings update: %s", feature)
        return jsonify({"success": False, "error": f"Unsupported feature: {feature}"})
    except Exception as e:
        logger.error("Error updating settings: %s", str(e))
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/system_settings', methods=['POST'])
def update_system_settings():
    try:
        global log_folder, log_file_name
        data = request.json
        
        if 'log_folder' in data:
            log_folder = data['log_folder']
            os.makedirs(log_folder, exist_ok=True)
        if 'log_name' in data:
            log_file_name = f"{data['log_name']}.json" if data['log_name'] else "detection_log.json"
        
        logger.info("Updated system settings: log_folder=%s, log_file_name=%s", log_folder, log_file_name)
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Error updating system settings: %s", str(e))
        return jsonify({"success": False, "error": str(e)})

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    socketio.emit('status_update', {'message': 'Connected to server'})
    for feature, active in active_detectors.items():
        socketio.emit('status_update', {'feature': feature, 'active': active})

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
        logger.error("Error queuing frame: %s", str(e))

if __name__ == '__main__':
    if load_models():
        threading.Thread(target=process_frame_worker, daemon=True).start()
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to start server due to model loading failure")