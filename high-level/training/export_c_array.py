
from pathlib import Path

MODEL_PATH = "product_classifier_int8.tflite"
OUT_PATH = "model_data.h"
VAR_NAME = "g_product_classifier_model"

data = Path(MODEL_PATH).read_bytes()

with open(OUT_PATH, "w") as f:
    f.write("//product_classifier_int8.tflite output\n")
    f.write("#pragma once\n\n")
    f.write("#include <cstdint>\n\n")
    f.write(f"alignas(16) const unsigned char {VAR_NAME}[] = {{\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        f.write(f"    {line},\n")

    f.write("};\n\n")
    f.write(f"const unsigned int {VAR_NAME}_len = {len(data)};\n")

print(f"Saved -> {OUT_PATH}")
print(f"Model size: {len(data)} bytes ({len(data)/1024:.1f} KB)")

