# import time
# import collections
# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# from scipy import stats

# # Load the model
# model = load_model('/home/sanjib/Desktop/backend/src/word/collective2.h5')
# actions = np.array( [
#     "always", "ask", "bathroom", "bird", "black", "blue", "book", "brown", "busy", "buy",
#     "candy", "car", "cat", "clean", "come", "cook", "deaf", "draw", "drink", "eat",
#     "fine", "finish", "forget", "give", "go", "good", "green", "happy", "hello", "help",
#     "house", "how", "hungry", "i", "icecream", "know", "learn", "like", "love_it", "man",
#     "milk", "more", "name", "never", "no", "not", "pay_attention", "play", "please", "red",
#     "right", "room", "sad", "same", "say", "see", "shhh", "sleep", "sorry", "test", "text",
#     "thankyou", "time", "today", "tomorrow", "understand", "walk", "want", "water", "what",
#     "where", "white", "who", "woman", "work", "write", "wrong", "yes", "yesterday", "you"
# ]) 


# colors = []
# # create colors for each action
# for i in range(len(actions)):
#     colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         # print(num, prob)
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
#     return output_frame

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# filtered_hand = list(range(21))
# filtered_pose = [11, 12, 13, 14, 15, 16]
# filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
#                  61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
#                  107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
#                  155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 215,
#                  234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
#                  293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
#                  332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
#                  381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
#                  415, 435, 454, 466]

# def filter_landmarks(landmarks, indices):
#     """Return a list of only selected landmarks."""
#     if landmarks is None:
#         return None
#     from mediapipe.framework.formats import landmark_pb2
#     filtered = landmark_pb2.NormalizedLandmarkList()
#     for i in indices:
#         if i < len(landmarks.landmark):
#             filtered.landmark.append(landmarks.landmark[i])
#     return filtered

# def mediapipe_detection(image, model):
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   image.flags.writeable = False
#   results = model.process(image)
#   image.flags.writeable = True
#   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#   return image, results

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.face_landmarks, filtered_face))
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.pose_landmarks, filtered_pose))
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.left_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.right_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS)

# def draw_styled_landmarks(image, results):
#     # Face
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.face_landmarks, filtered_face), connections = None,
#         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
#     )
#     # Pose
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.pose_landmarks, filtered_pose), connections = None,
#         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
#     )
#     # Left hand
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.left_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS,
#         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
#         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
#     )
#     # Right hand
#     mp_drawing.draw_landmarks(
#         image, filter_landmarks(results.right_hand_landmarks, filtered_hand), mp_holistic.HAND_CONNECTIONS,
#         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
#     )


# def extract_keypoints(results):
#     # Pose
#     if results.pose_landmarks:
#         pose = [
#             [res.x, res.y, res.z, res.visibility]
#             for i, res in enumerate(results.pose_landmarks.landmark)
#             if i in filtered_pose
#         ]
#         pose = np.array(pose).flatten()
#     else:
#         pose = np.zeros(len(filtered_pose) * 4)

#     # Face
#     if results.face_landmarks:
#         face = [
#             [res.x, res.y, res.z]
#             for i, res in enumerate(results.face_landmarks.landmark)
#             if i in filtered_face
#         ]
#         face = np.array(face).flatten()
#     else:
#         face = np.zeros(len(filtered_face) * 3)

#     # Left hand
#     if results.left_hand_landmarks:
#         lh = [
#             [res.x, res.y, res.z]
#             for i, res in enumerate(results.left_hand_landmarks.landmark)
#             if i in filtered_hand
#         ]
#         lh = np.array(lh).flatten()
#     else:
#         lh = np.zeros(len(filtered_hand) * 3)

#     # Right hand
#     if results.right_hand_landmarks:
#         rh = [
#             [res.x, res.y, res.z]
#             for i, res in enumerate(results.right_hand_landmarks.landmark)
#             if i in filtered_hand
#         ]
#         rh = np.array(rh).flatten()
#     else:
#         rh = np.zeros(len(filtered_hand) * 3)

#     return np.concatenate([pose, face, lh, rh])

# sequence = []
# sentence = []
# predictions = collections.deque(maxlen=20)
# threshold = 0.6

# # For FPS limiting
# prev_time = 0
# fps_limit = 10  # cap at ~10 FPS

# cap = cv2.VideoCapture(0)

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame")
#             break

#         current_time = time.time()
#         if current_time - prev_time < 1.0 / fps_limit:
#             continue
#         prev_time = current_time

#         # Step 1: Detection
#         image, results = mediapipe_detection(frame, holistic)
#         draw_styled_landmarks(image, results)

#         # Step 2: Extract keypoints
#         keypoints = extract_keypoints(results)
#         if np.count_nonzero(keypoints) < 10:
#             # Skip frame if too many keypoints are missing
#             continue

#         sequence.append(keypoints)
#         sequence = sequence[-30:]

#         # Step 3: Predict if sequence ready
#         if len(sequence) == 30:
#             input_seq = np.expand_dims(sequence, axis=0)
#             res = model.predict(input_seq, verbose=0)[0]
#             predicted_class = np.argmax(res)
#             confidence = res[predicted_class]

#             predictions.append(predicted_class)

#             # Step 4: Smoothing logic
#             if predictions.count(predicted_class) > 12 and confidence > threshold:
#                 if not sentence or (actions[predicted_class] != sentence[-1]):
#                     sentence.append(actions[predicted_class])

#         # Step 5: Draw sentence and probabilities
#         image = prob_viz(res, actions, image, colors) if len(sequence) == 30 else image

#         cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence[-5:]), (3,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow('ASL Real-Time Detection', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# Required libraries:
# pip install opencv-python mediapipe tensorflow pyttsx3 google-generativeai

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import collections
import google.generativeai as genai
import tensorflow as tf

# --- Config ---
MODEL_PATH = '/home/sanjib/Desktop/backend/src/word/collective2.h5'
GEMINI_API_KEY = 'AIzaSyA6nfRukBDIiD7xGBkL58slZqtwAfXcgqA'
FPS_LIMIT = 15
CONFIDENCE_THRESHOLD = 0.65

# --- Initialization ---
model = tf.keras.models.load_model(MODEL_PATH)
actions = np.array( [
    "always", "ask", "bathroom", "bird", "black", "blue", "book", "brown", "busy", "buy",
    "candy", "car", "cat", "clean", "come", "cook", "deaf", "draw", "drink", "eat",
    "fine", "finish", "forget", "give", "go", "good", "green", "happy", "hello", "help",
    "house", "how", "hungry", "i", "icecream", "know", "learn", "like", "love_it", "man",
    "milk", "more", "name", "never", "no", "not", "pay_attention", "play", "please", "red",
    "right", "room", "sad", "same", "say", "see", "shhh", "sleep", "sorry", "test", "text",
    "thankyou", "time", "today", "tomorrow", "understand", "walk", "want", "water", "what",
    "where", "white", "who", "woman", "work", "write", "wrong", "yes", "yesterday", "you"
])
tts = pyttsx3.init()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Generate colors for actions
colors = [(np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)) for _ in actions]

# --- Gemini Sentence Generation ---
def call_gemini(sentence):
    # prompt = f"Convert the following ASL signs into a grammatically correct sentence: {sentence}"
    prompt = f"""
        Objective:
        You have developed an isolated American Sign Language (ASL) word recognition model. After each run, the model stores the recognized ASL words in a list. These words represent the user's intended message, but may not be in grammatically correct order or format. Your task is to convert this list into a simple, natural-sounding English sentence.

        Instructions:
        - Input: You will be provided with a Python list containing the recognized ASL words.
        - Processing:
        - Rearrange the words to form a grammatically correct and logically meaningful English sentence.
        - Apply common language transformations where applicable. Examples:
            - "how you" → "how are you"
            - "i fine" → "I am fine"
            - "you want water" → "Do you want water?"
            - "thankyou" → "Thank you"
            - "i hungry" → "I am hungry"
            - "go room" → "Go to the room"
            - "you where" → "Where are you?"
        - Capitalize the sentence and use appropriate punctuation.
        - Add missing linking words (like “is”, “are”, “am”, “to”, “do”) as needed to make the sentence natural.
        - Output: Generate a clear and concise English sentence that represents the intended meaning of the ASL word list. Don't add any things like here is the sentence.

        Considerations:
        - Simplicity: Keep the structure and vocabulary simple.
        - Clarity: Ensure the meaning is easy to understand.
        - Grammar: Maintain correct syntax, tense, and punctuation.
        - Naturalness: The sentence should sound like something a fluent speaker would say.

        Input word list:
        sentence = {sentence}
        """
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    output = response.text.strip()
    print("\n[Gemini Response]", response.text.strip())
    return response.text.strip()

# --- Text to Speech ---
def speak(text):
    tts.say(text)
    tts.runAndWait()

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Landmark filtering indices
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46,
                 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
                 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109,
                 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152,
                 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172,
                 173, 176, 178, 181, 185, 191, 215, 234, 246, 249, 251,
                 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
                 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362,
                 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384,
                 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405,
                 409, 415, 435, 454, 466]

# --- Helper Functions ---
def filter_landmarks(landmarks, indices):
    if landmarks is None: return None
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

# --- Real-time Loop ---
sequence, sentence = [], []
output = ''
predictions = collections.deque(maxlen=20)
prev_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        current_time = time.time()
        if current_time - prev_time < 1.0 / FPS_LIMIT:
            continue
        prev_time = current_time

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        if np.count_nonzero(keypoints) < 10:
            continue

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            input_seq = np.expand_dims(sequence, axis=0)
            res = model.predict(input_seq, verbose=0)[0]
            predicted_class = np.argmax(res)
            confidence = res[predicted_class]

            predictions.append(predicted_class)
            if predictions.count(predicted_class) > 12 and confidence > CONFIDENCE_THRESHOLD:
                if not sentence or actions[predicted_class] != sentence[-1]:
                    sentence.append(actions[predicted_class])

            # Display top 4 predictions
            top_indices = res.argsort()[-4:][::-1]
            for i, idx in enumerate(top_indices):
                cv2.putText(image, f"{actions[idx]}: {res[idx]*100:.1f}%", (30, 80 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[idx], 3)

        # Draw sentence on screen
        cv2.rectangle(image, (0, 0), (960, 50), (0, 0, 0), -1)
        cv2.putText(image, ' '.join(sentence[-6:]), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('DribyaDristi ASL Detection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
        elif key == ord('g'):
            if sentence:
                output = call_gemini(sentence)
                print("\n[Gemini Sentence]", output)
        elif key == ord('s'):
            print("Speaking the sentence...")
            if sentence and output:
                speak(output)
            elif sentence:
                speak(' '.join(sentence[-6:]))

cap.release()
cv2.destroyAllWindows()

