import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow.keras as keras
import time
from collections import deque
import pyttsx3
import threading

# Initialize webcam and detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    raise SystemExit
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = HandDetector(maxHands=2, detectionCon=0.7)
offset = 20
imgSize = 300

# Load the trained model
model_path = r"D:\Vision\GradProjrct\asl_model_300x300_1.h5" 
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit

# Define labels
labels = ['done1', 'done2', 'forget1', 'forget2', 'happy', 'help', 'hi', 'how are', 'i', 'love', 'need', 'nice', 'to meet', 'you']

# Prediction buffer and sequence tracking
prediction_buffer = deque(maxlen=5)
sequence_history = deque(maxlen=50)
sequence_timeout = 2.0
predicted_labels = []  # Store unique predicted labels

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Increase speech rate for faster delivery
engine.setProperty('volume', 1.0)

def speak_labels(labels_list):
    """Run speech for a list of labels as a single continuous phrase."""
    print(f"Speaking labels: {labels_list}")  # Debug print
    # Concatenate labels into a single string with minimal spacing
    combined_text = " ".join(labels_list).replace("  ", " ")  # Ensure single spaces
    engine.say(combined_text)
    engine.runAndWait()

def preprocess_image(img, x, y, w, h, imgSize, offset):
    imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
    if imgCrop.size == 0:
        return None
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    try:
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(w * k)
            if wCal > imgSize:
                wCal = imgSize
                k = imgSize / w
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize[:, :wCal]
        else:
            k = imgSize / w
            hCal = int(h * k)
            if hCal > imgSize:
                hCal = imgSize
                k = imgSize / h
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize[:hCal, :]
        return imgWhite
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

try:
    prev_time = time.time()
    last_predicted_label = None
    min_time_gap = 1.0
    last_prediction_time = 0.0
    displayed_text = ""
    was_hand_detected = False
    no_hand_start_time = None  # Track when no hands are first detected
    no_hand_delay = 1.0  # Require 1 second of no hands before speaking

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        hands, img = detector.findHands(img)
        hand_count = len(hands)
        info = f"Hands detected: {hand_count}"

        if hands:
            imgWhite = None
            if hand_count == 1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = preprocess_image(img, x, y, w, h, imgSize, offset)
                info += " (1 hand)"
            elif hand_count == 2:
                x1, y1, w1, h1 = hands[0]['bbox']
                x2, y2, w2, h2 = hands[1]['bbox']
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min
                imgWhite = preprocess_image(img, x, y, w, h, imgSize, offset)
                info += " (2 hands)"

            if imgWhite is not None:
                cv2.imshow("ImageWhite", imgWhite)
                imgProcessed = imgWhite.astype('float32') / 255.0
                imgProcessed = np.expand_dims(imgProcessed, axis=0)
                prediction = model.predict(imgProcessed, verbose=0)[0]
                prediction_buffer.append(prediction)
                was_hand_detected = True
                no_hand_start_time = None  # Reset timer when hands are detected

        else:
            # No hands detected
            prediction_buffer.clear()
            blank_img = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            cv2.imshow("ImageWhite", blank_img)
            if was_hand_detected:
                if no_hand_start_time is None:
                    no_hand_start_time = time.time()  # Start timer
                elif time.time() - no_hand_start_time >= no_hand_delay:
                    if predicted_labels:
                        labels_to_speak = predicted_labels.copy()
                        threading.Thread(target=speak_labels, args=(labels_to_speak,), daemon=True).start()
                        predicted_labels.clear()
                        sequence_history.clear()
                        last_predicted_label = None
                        displayed_text = ""
                    was_hand_detected = False
                    no_hand_start_time = None
            print("No hands detected, predicted_labels:", predicted_labels)  # Debug print

        if len(prediction_buffer) == 5:
            avg_prediction = np.mean(prediction_buffer, axis=0)
            predicted_class = np.argmax(avg_prediction)
            confidence = avg_prediction[predicted_class]
            model_pred = labels[predicted_class]

            display_label = None
            current_time = time.time()

            if model_pred != last_predicted_label or (current_time - last_prediction_time) > min_time_gap:
                if confidence > 0.7:
                    sequence_history.append((model_pred, current_time))

                    # forget1 -> forget2
                    forget1_time = None
                    for label, t in sequence_history:
                        if label == 'forget1':
                            forget1_time = t
                        elif label == 'forget2' and forget1_time and t > forget1_time:
                            display_label = 'forget'
                            sequence_history.clear()
                            break

                    # done1 -> done2
                    if not display_label:
                        done1_time = None
                        for label, t in sequence_history:
                            if label == 'done1':
                                done1_time = t
                            elif label == 'done2' and done1_time and t > done1_time:
                                display_label = 'done'
                                sequence_history.clear()
                                break

                    

                    if not display_label and model_pred in [ 'happy', 'help', 'hi', 'how are', 'i', 'love', 'need','nice','to meet','you']:
                        display_label = model_pred

                    if display_label and confidence > 0.7:
                        info += f" | Predicted: {display_label} ({confidence:.2f})"
                        if display_label not in predicted_labels:
                            predicted_labels.append(display_label)
                            print(f"Added to predicted_labels: {display_label}")  # Debug print
                        displayed_text = " ".join(predicted_labels)
                        last_predicted_label = model_pred
                        last_prediction_time = current_time

        elif len(prediction_buffer) > 0:
            info += " | Collecting predictions..."

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"

        # Show info and displayed text
        cv2.putText(img, info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, fps_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if displayed_text:
            cv2.putText(img, f"Predicted: {displayed_text}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
