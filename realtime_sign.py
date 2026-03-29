import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. LOAD AI BRAIN ---
model = tf.keras.models.load_model('signspeak_model.keras')
label_map = np.load('label_map.npy', allow_pickle=True).item()
inv_label_map = {v: k for k, v in label_map.items()}

# --- 2. SETUP HAND LANDMARKER (TASKS API) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_speech_time = 0

print("🎥 Webcam started! Show your hand to the camera.")
print("⌨️ Press 'q' on your keyboard to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    # Convert OpenCV BGR to MediaPipe RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect landmarks
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            # --- 3. LANDMARK NORMALIZATION ---
            wrist = hand_lms[0]
            coords = []
            for lm in hand_lms:
                coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            # Zoom/Scaling normalization (Matches your training logic)
            max_val = max(max(coords), abs(min(coords)))
            if max_val == 0: max_val = 1
            normalized = [c / max_val for c in coords]

            # --- 4. PREDICTION ---
            prediction = model.predict(np.array([normalized]), verbose=0)
            char_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.90:
                letter = inv_label_map[char_id]
                cv2.putText(frame, f"Letter: {letter} ({int(confidence*100)}%)", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                
                # --- 5. TEXT-TO-SPEECH ---
                if time.time() - last_speech_time > 3:
                    try:
                        tts = gTTS(text=f"The letter is {letter}", lang='en')
                        tts.save("speech.mp3")
                        os.system("start speech.mp3")
                        last_speech_time = time.time()
                    except Exception as e:
                        print(f"Audio error: {e}")

    cv2.imshow("SignSpeak AI - Real Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()