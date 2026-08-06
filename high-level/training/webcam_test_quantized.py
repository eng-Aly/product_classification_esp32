"""
Live webcam test for product_classifier_int8.tflite — validates the
quantized model behaves like the float version before moving to ESP-DL export.

pip install opencv-python tensorflow

Controls:
  q - quit
"""

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "product_classifier_int8.tflite"
IMG_SIZE = 96
CLASS_NAMES = ["class_a", "class_b"]  # <-- set to your actual folder names, alphabetical order

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero_point = input_details["quantization"]
out_scale, out_zero_point = output_details["quantization"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0). Try a different index if you have multiple cameras.")

print("Press 'q' to quit.")
print(f"Input quant: scale={in_scale}, zero_point={in_zero_point}")
print(f"Output quant: scale={out_scale}, zero_point={out_zero_point}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    # BGR -> RGB, resize
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    # Replicate MobileNetV2 preprocess_input (scales pixel [0,255] -> [-1, 1])
    # then quantize that float range into int8 using the model's own scale/zero_point.
    normalized = (resized / 127.5) - 1.0  # matches mobilenet_v2.preprocess_input
    quantized = np.round(normalized / in_scale + in_zero_point).astype(np.int8)
    batch = np.expand_dims(quantized, axis=0)

    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # Dequantize int8 output back to float probabilities
    probs = (output.astype(np.float32) - out_zero_point) * out_scale

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label = CLASS_NAMES[class_idx]

    text = f"{label}: {confidence*100:.1f}%"
    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    for i, name in enumerate(CLASS_NAMES):
        line = f"{name}: {probs[i]*100:.1f}%"
        cv2.putText(frame, line, (20, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Quantized Model Test - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()