import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def get_hand_crop(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        h, w, _ = image.shape
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * w) - 20
            xmax = int(max(x_coords) * w) + 20
            ymin = int(min(y_coords) * h) - 20
            ymax = int(max(y_coords) * h) + 20
            crop = image[ymin:ymax, xmin:xmax]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return crop, image
    return None, image