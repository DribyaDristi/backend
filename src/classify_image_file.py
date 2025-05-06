import sys
from classify import classify_static_image_file

if len(sys.argv) != 2:
    print("Usage: python classify_image_file.py <image_path>")
    exit(1)

image_path = sys.argv[1]
try:
    result = classify_static_image_file(image_path)
    print(f"Predicted sign: {result}")
except Exception as e:
    print("Error:", e)