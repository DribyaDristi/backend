import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# 1) MODEL AND CLASS DECODING

def decode(value):
    class_indices = {
        'A': 0,  'B': 1,  'C': 2,   'D': 3,  'E': 4,   'F': 5,  'G': 6,
        'H': 7,  'I': 8,  'J': 9,   'K': 10, 'L': 11,  'M': 12, 'N': 13,
        'O': 14, 'P': 15, 'Q': 16,  'R': 17, 'S': 18,  'T': 19, 'U': 20,
        'V': 21, 'W': 22, 'X': 23,  'Y': 24, 'Z': 25, 'del': 26,
        'nothing': 27, 'space': 28
    }
    inv_map = {v: k for k, v in class_indices.items()}
    return inv_map.get(value, "Unknown")


model_path = "/home/sanjib/Desktop/backend/newmodel/Dribya_Dristi.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)


# 2) MEDIAPIPE HANDS DETECTION SETUP 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# 3) DEBOUNCER CLASS (locks in a stable label)

class Debouncer:
    def __init__(self, threshold=1.0):
        self.last_label = None
        self.last_time = 0.0
        self.threshold = threshold

    def update(self, current_label):
        now = time.time()
        if current_label != self.last_label:
            self.last_label = current_label
            self.last_time = now
            return None
        else:
            if now - self.last_time >= self.threshold:
                self.last_time = now
                return current_label
        return None



# 4) UTILITY: CROP HAND TO A SQUARE REGION + RESIZE TO 48×48

def pad_to_square(img, fill_color=(0, 0, 0)):
    """
    Given a BGR image, pad it to a square by adding black pixels
    so that width == height without distortion.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    padded = np.full((size, size, 3), fill_color, dtype=np.uint8)
    # Center the original img in this square:
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return padded


def crop_and_preprocess(image, hand_landmarks):
    h, w, _ = image.shape

    x_coords = []
    y_coords = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * w)
        py = int(lm.y * h)
        x_coords.append(px)
        y_coords.append(py)

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    box_size = max(xmax - xmin, ymax - ymin)
    box_size = int(box_size * 1.25)

    roi_x = max(0, center_x - box_size // 2)
    roi_y = max(0, center_y - box_size // 2)
   
    box_size = min(box_size, w - roi_x, h - roi_y)
    roi = image[roi_y : roi_y + box_size, roi_x : roi_x + box_size]

    square = pad_to_square(roi)
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 48, 48, 1), (roi_x, roi_y, box_size)



# REAL‐TIME CALLBACK

out = cv2.VideoWriter('output.mp4', -1, 30.0, (640,480))
cap = cv2.VideoCapture(0)
debouncer = Debouncer(threshold=1.0)

# Buffers to build words/sentence:
letter_buffer = []    
sentence = []         

print("▶️  Press 'q' to quit. Press 'c' to clear the entire sentence.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # frame = cv2.flip(frame, 1)  
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_label = None
    current_conf = 0.0
    bbox_coords = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            preprocessed, (rx, ry, rsize) = crop_and_preprocess(frame, hand_landmarks)
            bbox_coords = (rx, ry, rsize)

            preds = model.predict(preprocessed, verbose=0)[0]  # shape = (29,)
            idx = np.argmax(preds)
            current_conf = float(preds[idx] * 100)
            raw_label = decode(idx)

            # If confidence <30%, treat as "nothing":
            if current_conf < 30:
                current_label = "nothing"
            else:
                current_label = raw_label

        except Exception as e:
            current_label = None

    confirmed = None
    if current_label is not None:
        confirmed = debouncer.update(current_label)

    if confirmed is not None:
        if confirmed == "space":
            word = "".join(letter_buffer) if letter_buffer else ""
            if word:
                sentence.append(word)
            letter_buffer = []

        elif confirmed == "del":
            if letter_buffer:
                letter_buffer.pop()
        elif confirmed == "nothing":
            pass
        else:
            letter_buffer.append(confirmed)

        debouncer.last_label = None
        debouncer.last_time = time.time()

    if bbox_coords is not None:
        rx, ry, rsize = bbox_coords
        cv2.rectangle(
            frame,
            (rx, ry),
            (rx + rsize, ry + rsize),
            (255, 0, 0),
            2
        )

    if current_label is not None and bbox_coords is not None:
        text = f"{current_label} ({current_conf:.1f}%)"
        cv2.putText(
            frame,
            text,
            (bbox_coords[0], max(15, bbox_coords[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    full_sentence = " ".join(sentence + (["".join(letter_buffer)] if letter_buffer else []))

    cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        full_sentence,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    out.write(frame)
    cv2.imshow("ASL Real-Time Word Builder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c"):
        letter_buffer = []
        sentence = []
        print("🗑️  Cleared entire sentence.")

cap.release()
out.release()
cv2.destroyAllWindows()
