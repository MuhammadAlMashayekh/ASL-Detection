ASL Sign Language Recognition (Offline, Webcam-Based)

This project is a real-time American Sign Language (ASL) recognition system built with Python.
It uses a webcam, computer vision, and a trained deep learning model to recognize ASL signs and speak them aloud.

Features: 
1- Hand Detection → Detects one or two hands using cvzone.HandTrackingModule (MediaPipe-based).
2- Preprocessing → Crops and resizes the hand(s) into a fixed 300x300 canvas for model input.
3- Deep Learning Model → Loads a pre-trained Keras .h5 model to classify ASL signs.
4- Sequence Recognition → Combines multi-step signs:
  	a- done1 + done2 → done
    b- forget1 + forget2 → forget
5-Prediction Buffering → Averages predictions over 5 frames for smoother results.
6- Text-to-Speech → Uses pyttsx3 to speak recognized words offline.
7- Offline Operation → Runs entirely on your machine, no internet required.

Supported Signs:
The system can recognize these labels: done1, done2, forget1, forget2, happy, help, hi,
how are, i, love, need, nice, to meet, you.
The program merges them into natural words/phrases:
done1 + done2 → "done"
forget1 + forget2 → "forget"
Others (e.g., love, hi, help) are recognized directly.

Requirements:
Install dependencies with:
pip install opencv-python cvzone numpy tensorflow pyttsx3

System Flow:
1- Capture frame from webcam
2- Detect hands and crop region of interest
3- Preprocess (resize, center, normalize)
4- Predict sign using trained model
5- Smooth predictions with buffer
6- Combine multi-step signs into full words
7- Display on screen + speak aloud
