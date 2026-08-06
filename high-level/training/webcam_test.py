

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = "product_classifier.keras"
IMG_SIZE = 96
CLASS_NAMES = ["class_a", "class_b"]  # <-- set to your actual folder names, alphabetical order

model = tf.keras.models.load_model(MODEL_PATH)
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0). Try a different index if you have multiple cameras.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    # BGR (OpenCV) -> RGB (model expects RGB, matching image_dataset_from_directory)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    batch = preprocess(batch)

    preds = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = CLASS_NAMES[class_idx]

    # Overlay prediction on the frame
    text = f"{label}: {confidence*100:.1f}%"
    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # orange if low confidence
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Show all class probabilities in a corner, useful for spotting borderline cases
    for i, name in enumerate(CLASS_NAMES):
        line = f"{name}: {preds[i]*100:.1f}%"
        cv2.putText(frame, line, (20, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Product Classifier - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
