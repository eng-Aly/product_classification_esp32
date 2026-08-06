
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODEL_PATH = "product_classifier.keras"
output_file = os.path.splitext(MODEL_PATH)[0] + "_quantization_info.txt"
IMG_SIZE = 96
NUM_CALIBRATION_SAMPLES = 100  

model = tf.keras.models.load_model(MODEL_PATH)
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input


def representative_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=True,
        seed=123,
    )
    count = 0
    for images, _ in ds:
        if count >= NUM_CALIBRATION_SAMPLES:
            break
        img = preprocess(images)
        yield [img]
        count += 1

# ---- Convert with full int8 quantization ----
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force full int8 (weights + activations), not just weights.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

out_path = "product_classifier_int8.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_model)

print(f"Saved -> {out_path}")
print(f"Size: {len(tflite_model) / 1024:.1f} KB")


interpreter = tf.lite.Interpreter(model_path=out_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(output_file, "w") as f:
    f.write(f"Quantized model input dtype: {input_details[0]['dtype']}\n")
    f.write(f"Input quantization (scale, zero_point): {input_details[0]['quantization']}\n")
    f.write(f"Output quantization (scale, zero_point): {output_details[0]['quantization']}\n\n")
    f.write("IMPORTANT:\n")
    f.write("Note these scale/zero_point values. You'll need them\n")
    f.write("on the ESP32 side to convert raw camera pixels into the int8 range\n")
    f.write("the model expects, and to decode int8 outputs back into probabilities.\n")

print(f"Quantization information saved to {output_file}")