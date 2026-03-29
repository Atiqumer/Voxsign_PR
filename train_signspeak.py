import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- 1. SETTINGS ---
# Since you already unzipped, we point directly to the folder
DATASET_PATH = 'dataset' 
TRAIN_DIR = os.path.join(DATASET_PATH, 'Train_Alphabet')
MODEL_NAME = 'signspeak_model.keras'

# --- 2. LANDMARK EXTRACTION SETUP ---
# Ensure hand_landmarker.task is in Desktop\Don
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

X_data, y_data = [], []

# Map folder names to numeric IDs
classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
label_map = {name: i for i, name in enumerate(classes)}

print(f"✅ Found {len(classes)} alphabet classes.")
print("⚡ Starting landmark extraction (Location & Zoom normalized)...")

# --- 3. THE EXTRACTION LOOP ---
for class_name, label_id in label_map.items():
    class_folder = os.path.join(TRAIN_DIR, class_name)
    images = os.listdir(class_folder)[:200] # Standardizing to 200 images per letter
    
    print(f"Processing class: {class_name}...")
    for img_name in images:
        try:
            image_path = os.path.join(class_folder, img_name)
            image = mp.Image.create_from_file(image_path)
            result = detector.detect(image)

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]
                wrist = lm_list[0]
                
                # Normalization Step 1: Subtract Wrist (makes it location-independent)
                temp_coords = []
                for lm in lm_list:
                    temp_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                # Normalization Step 2: Scale/Zoom (makes it distance-independent)
                max_val = max(max(temp_coords), abs(min(temp_coords)))
                if max_val == 0: max_val = 1 # Prevent division by zero
                normalized = [c / max_val for c in temp_coords]
                
                X_data.append(normalized)
                y_data.append(label_id)
        except Exception:
            continue

X, y = np.array(X_data), np.array(y_data)
np.save('label_map.npy', label_map) # Save labels for the real-time script

# --- 4. BUILD & TRAIN THE BRAIN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)), # 21 landmarks * (x,y,z)
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Training SignSpeak AI Brain...")
model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# --- 5. SAVE ---
model.save(MODEL_NAME)
print(f"\n✅ SUCCESS: Model saved as {MODEL_NAME}")