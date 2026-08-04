import cv2
import os


from pathlib import Path

# Directory to save images
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR.parent / "data" / "class1"

os.makedirs(SAVE_DIR, exist_ok=True)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 's' to save an image.")
print("Press 'q' to quit.")

img_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = os.path.join(SAVE_DIR, f"image_{img_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        img_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
