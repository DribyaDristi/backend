# import cv2
# from hand_detector import get_hand_crop
# from classify import classify_image
# import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture("http://192.168.254.30:8080/video")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cropped_hand, output_frame = get_hand_crop(frame)
#     # plt.
#     # plt.imshow(frame)

#     # show the cropped hand image
#     # if cropped_hand is not None:
#     #     plt.imshow(output_frame)
#     #     plt.axis('off')
#     #     plt.show()

#     if cropped_hand is not None:
#         try:
#             predicted_sign = classify_image(cropped_hand)
#             cv2.putText(output_frame, predicted_sign, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
#         except Exception as e:
#             print("Prediction error:", e)

#     cv2.imshow("ASL Sign Detection", output_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from hand_detector import get_hand_crop
from classify import classify_image
import numpy as np
import time

# Initialize variables for stability detection
previous_positions = []
stability_threshold = 60  # pixel distance threshold
stability_frames = 20     # number of frames to check
last_prediction_time = 0
prediction_cooldown = 3   # seconds between predictions
last_prediction = None
prediction_display_duration = 3  # seconds to show prediction

cap = cv2.VideoCapture(0)

# ...existing code for helper functions...
def calculate_hand_center(cropped_hand):
    if cropped_hand is None:
        return None
    h, w = cropped_hand.shape[:2]
    return (w//2, h//2)

def is_hand_stable(positions, threshold):
    if len(positions) < stability_frames:
        return False
    
    # Calculate maximum distance between any two points in recent positions
    max_distance = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                         (positions[i][1] - positions[j][1])**2)
            max_distance = max(max_distance, dist)
    
    return max_distance < threshold


while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped_hand, output_frame = get_hand_crop(frame)
    current_time = time.time()
    
    if cropped_hand is not None:
        hand_center = calculate_hand_center(cropped_hand)
        if hand_center:
            previous_positions.append(hand_center)
            previous_positions = previous_positions[-stability_frames:]
            
            if (is_hand_stable(previous_positions, stability_threshold) and 
                current_time - last_prediction_time > prediction_cooldown):
                try:
                    predicted_sign = classify_image(cropped_hand)
                    last_prediction = (predicted_sign, current_time)
                    last_prediction_time = current_time
                except Exception as e:
                    print("Prediction error:", e)
            else:
                cv2.putText(output_frame, "Stabilizing...", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display prediction if within display duration
    if last_prediction and (current_time - last_prediction[1] <= prediction_display_duration):
        cv2.putText(output_frame, last_prediction[0], (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("ASL Sign Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()