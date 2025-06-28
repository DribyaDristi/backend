import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf
from scipy import stats
# Load the model
model = load_model('/home/notme/Programs/backend/src/word/test.h5')
actions = np.array(['book','buy','candy','drink','eat','hungry','icecream','like','milk','more','no','not','right','want','wrong','yes']) 

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

colors = []
# create colors for each action
for i in range(len(actions)):
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # print(num, prob)
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
                 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
                 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 215,
                 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
                 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                 415, 435, 454, 466]

def filter_landmarks(landmarks, indices):
    """Return a list of only selected landmarks."""
    if landmarks is None:
        return None
    from mediapipe.framework.formats import landmark_pb2
    filtered = landmark_pb2.NormalizedLandmarkList()
    for i in indices:
        if i < len(landmarks.landmark):
            filtered.landmark.append(landmarks.landmark[i])
    return filtered

def mediapipe_detection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = model.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.face_landmarks, filtered_face))
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.pose_landmarks, filtered_pose))
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.left_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.right_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Face
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.face_landmarks, filtered_face), connections = None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    )
    # Pose
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.pose_landmarks, filtered_pose), connections = None,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
    )
    # Left hand
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.left_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
    )
    # Right hand
    mp_drawing.draw_landmarks(
        image, filter_landmarks(results.right_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
    )


def extract_keypoints(results):
    # Pose
    if results.pose_landmarks:
        pose = [
            [res.x, res.y, res.z, res.visibility]
            for i, res in enumerate(results.pose_landmarks.landmark)
            if i in filtered_pose
        ]
        pose = np.array(pose).flatten()
    else:
        pose = np.zeros(len(filtered_pose) * 4)

    # Face
    if results.face_landmarks:
        face = [
            [res.x, res.y, res.z]
            for i, res in enumerate(results.face_landmarks.landmark)
            if i in filtered_face
        ]
        face = np.array(face).flatten()
    else:
        face = np.zeros(len(filtered_face) * 3)

    # Left hand
    if results.left_hand_landmarks:
        lh = [
            [res.x, res.y, res.z]
            for i, res in enumerate(results.left_hand_landmarks.landmark)
            if i in filtered_hand
        ]
        lh = np.array(lh).flatten()
    else:
        lh = np.zeros(len(filtered_hand) * 3)

    # Right hand
    if results.right_hand_landmarks:
        rh = [
            [res.x, res.y, res.z]
            for i, res in enumerate(results.right_hand_landmarks.landmark)
            if i in filtered_hand
        ]
        rh = np.array(rh).flatten()
    else:
        rh = np.zeros(len(filtered_hand) * 3)

    return np.concatenate([pose, face, lh, rh])


cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))


        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            cv2.putText(image, f"Predicted: {actions[np.argmax(res)]} {res[np.argmax(res)]*100}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()