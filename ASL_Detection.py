from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import time
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Initialize HandDetector with tuned parameters
detector = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.7)

# Load your trained model
try:
    model = load_model("C:/Users/mrmoh/Desktop/app/Backend/asl.h5")
    print("✅ Model loaded successfully")
    print("Model summary:")
    print(model.summary())
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class_names = ['done1', 'done2', 'forget1', 'forget2', 'happy', 'help', 'hi', 'how are', 'i', 'love', 'need', 'nice', 'to meet', 'you']

# Store connected clients
connected_clients = {}

def preprocess_image(img, x, y, w, h, imgSize=300, offset=20):
    """Preprocess hand image for model prediction"""
    try:
        # Crop hand region with offset
        imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
        if imgCrop.size == 0:
            print("Error: Empty cropped image")
            return None

        # Calculate aspect ratio and resize while preserving it
        aspectRatio = h / w
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        if aspectRatio > 1:
            # Height > Width: Fit to height, center horizontally
            k = imgSize / h
            wCal = min(int(w * k), imgSize)  # Ensure width doesn't exceed imgSize
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            # Width >= Height: Fit to width, center vertically
            k = imgSize / w
            hCal = min(int(h * k), imgSize)  # Ensure height doesn't exceed imgSize
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Save processed image for debugging (first few frames)
        if not hasattr(preprocess_image, 'save_count'):
            preprocess_image.save_count = 0
        if preprocess_image.save_count < 5:  # Save first 5 processed images
            cv2.imwrite(f"debug_processed_hand_{preprocess_image.save_count}.jpg", imgWhite)
            print(f"Saved debug image: debug_processed_hand_{preprocess_image.save_count}.jpg")
            preprocess_image.save_count += 1

        return imgWhite
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

def preprocess_hand(hand_img):
    """Preprocess hand image for model prediction"""
    try:
        hand_img = hand_img.astype(np.float32) / 255.0
        hand_img = img_to_array(hand_img)
        hand_img = np.expand_dims(hand_img, axis=0)
        return hand_img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def detect_hands(frame):
    """Detect hands using cvzone HandDetector with frame preprocessing"""
    try:
        # Ensure frame is in BGR format
        if frame.shape[2] != 3:
            print("Invalid frame: Expected 3 channels (BGR)")
            return None, None, 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR if needed
        
        # Adjust brightness and contrast
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Log frame properties
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, mean pixel: {np.mean(frame):.1f}, min: {np.min(frame)}, max: {np.max(frame)}")
        
        # Save a sample frame with bounding boxes for inspection
        debug_frame = frame.copy()
        hands, _ = detector.findHands(frame, draw=False)  # Disable drawing on main frame
        if hands:
            if len(hands) == 1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                score = hand.get('score', [0.0])[0] if 'score' in hand else 0.0
                print(f"Single hand detected: x={x}, y={y}, w={w}, h={h}, confidence={score:.2f}")
                # Draw bounding box on debug frame
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # Two hands
                x1, y1, w1, h1 = hands[0]['bbox']
                x2, y2, w2, h2 = hands[1]['bbox']
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                scores = [hand.get('score', [0.0])[0] for hand in hands]
                print(f"Two hands detected: combined bbox x={x}, y={y}, w={w}, h={h}, confidences={scores}")
                # Draw bounding boxes on debug frame
                cv2.rectangle(debug_frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                cv2.rectangle(debug_frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            hand_roi = preprocess_image(frame, x, y, w, h, imgSize=300, offset=20)
            if hand_roi is None:
                print("No valid hand ROI")
                return None, None, 0
            print(f"Hand ROI shape: {hand_roi.shape}")
            
            # Save debug frame with bounding boxes (first few frames)
            if not hasattr(detect_hands, 'frame_count'):
                detect_hands.frame_count = 0
            if detect_hands.frame_count < 5:  # Save first 5 frames
                cv2.imwrite(f"debug_frame_with_bbox_{detect_hands.frame_count}.jpg", debug_frame)
                print(f"Saved debug frame: debug_frame_with_bbox_{detect_hands.frame_count}.jpg")
                detect_hands.frame_count += 1
                
            return hand_roi, (x, y, w, h), w * h
        print("No hands detected")
        return None, None, 0
    except Exception as e:
        print(f"Error in hand detection: {e}")
        return None, None, 0

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': time.time(),
        'predictions_count': 0
    }
    print(f"🔗 Client {client_id} connected. Total clients: {len(connected_clients)}")
    emit('connection_status', {'status': 'connected', 'message': 'Successfully connected to ASL detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in connected_clients:
        client_info = connected_clients[client_id]
        session_duration = time.time() - client_info['connected_at']
        print(f"🔌 Client {client_id} disconnected. Session duration: {session_duration:.1f}s, Predictions: {client_info['predictions_count']}")
        del connected_clients[client_id]

@socketio.on('video_frame')
def handle_video_frame(data):
    client_id = request.sid
    print(f"📷 Frame received from client {client_id}")
    try:
        if model is None:
            emit('prediction_result', {
                'error': 'Model not loaded',
                'status': 'error'
            })
            return

        image_data = data['image']
        encoded_data = image_data.split(',')[1] if ',' in image_data else image_data
        np_data = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        print(f"Decoded frame size: {len(np_data)} bytes")
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            print("Invalid frame received")
            emit('prediction_result', {
                'error': 'Invalid frame data',
                'status': 'error'
            })
            return

        # Resize frame to 640x480 to reduce processing load
        frame = cv2.resize(frame, (640, 480))

        # Log raw frame properties
        print(f"Raw frame shape: {frame.shape}, dtype: {frame.dtype}, mean pixel: {np.mean(frame):.1f}")

        hand_img, bbox, contour_area = detect_hands(frame)

        if hand_img is None:
            emit('prediction_result', {
                'label': 'No hand detected',
                'confidence': 0.0,
                'status': 'no_hand',
                'frame_size': frame.shape[:2]
            })
            return

        processed_img = preprocess_hand(hand_img)
        if processed_img is None:
            emit('prediction_result', {
                'error': 'Preprocessing failed',
                'status': 'error'
            })
            return

        prediction = model.predict(processed_img, verbose=0)[0]
        max_index = np.argmax(prediction)
        label = class_names[max_index]
        confidence = float(prediction[max_index])
        print(f"Predicted: {label} with confidence {confidence:.2f}, Top 3: {sorted([(class_names[i], float(prediction[i])) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)[:3]}")

        if client_id in connected_clients:
            connected_clients[client_id]['predictions_count'] += 1

        emit('prediction_result', {
            'label': label,
            'confidence': confidence,
            'status': 'success',
            'bbox': bbox,
            'contour_area': int(contour_area),
            'hand_size': hand_img.shape[:2],
            'frame_size': frame.shape[:2],
            'all_predictions': [
                {
                    'class': class_names[i],
                    'confidence': float(prediction[i])
                } for i in range(len(class_names))
            ]
        })

    except Exception as e:
        print(f"❌ Error processing frame for client {client_id}: {str(e)}")
        emit('prediction_result', {
            'error': str(e),
            'status': 'error'
        })

@socketio.on('get_server_stats')
def handle_get_stats():
    total_predictions = sum(client['predictions_count'] for client in connected_clients.values())
    emit('server_stats', {
        'connected_clients': len(connected_clients),
        'total_predictions': total_predictions,
        'available_classes': class_names,
        'model_loaded': model is not None
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'connected_clients': len(connected_clients)
    }

if __name__ == '__main__':
    print("🚀 Starting ASL Detection Server with SocketIO...")
    print("📋 Available classes:", class_names)
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)