# import cv2
# import mediapipe as mp
# import pandas as pd
# import numpy as np
# import time
# import tensorflow as tf
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Initialize the TensorFlow Lite interpreter
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
# pred_fn = interpreter.get_signature_runner("serving_default")

# # Load training data
# train = pd.read_csv("train.csv")
# train['sign_ord'] = train['sign'].astype('category').cat.codes
# SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
# ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# # Initialize MediaPipe holistic model
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# def create_frame_landmark_df(results, frame, xyz):
#     xyz_skel = (xyz[["type", "landmark_index"]].drop_duplicates().reset_index(drop=True).copy())
#     face = pd.DataFrame()
#     pose = pd.DataFrame()
#     left_hand = pd.DataFrame()
#     right_hand = pd.DataFrame()
    
#     if results.face_landmarks:
#         for i, point in enumerate(results.face_landmarks.landmark):
#             face.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
#     if results.pose_landmarks:
#         for i, point in enumerate(results.pose_landmarks.landmark):
#             pose.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
#     if results.left_hand_landmarks:
#         for i, point in enumerate(results.left_hand_landmarks.landmark):
#             left_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
#     if results.right_hand_landmarks:
#         for i, point in enumerate(results.right_hand_landmarks.landmark):
#             right_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    
#     face = face.reset_index().rename(columns={"index": "landmark_index"}).assign(type="face")
#     pose = pose.reset_index().rename(columns={"index": "landmark_index"}).assign(type="pose")
#     left_hand = left_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="left_hand")
#     right_hand = right_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="right_hand")
    
#     landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
#     landmarks = xyz_skel.merge(landmarks, on=["type", "landmark_index"], how="left").assign(frame=frame)
    
#     return landmarks

# def get_display_message_from_api(recognised_words):
#     GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
#     genai.configure(api_key=GOOGLE_API_KEY) 
    
#     model = genai.GenerativeModel('gemini-pro')
    
#     prompt = f"""
#             Objective:
#             You have developed an isolated American Sign Language (ASL) word recognition model. At the end of each run, the model stores the recognized words in a list. However, the words may not necessarily be in the correct order. Your objective is to utilize these recognized words to construct a coherent and meaningful English sentence. The resulting sentence should be as simple as possible while still accurately conveying the intended meaning.

#             Instructions:

#             - Input: You will be provided with a Python list containing the recognized ASL words from your model. The contents of this list may vary depending on the output of your model.
#             - Processing: Rearrange the words in the list to form a grammatically correct and logically valid English sentence. Take into consideration the context and logical flow of the sentence. Always ignore the word "TV".
#             - Output: Generate a concise English sentence that accurately conveys the meaning of the recognized ASL words.

#             Considerations:

#             - Simplicity: Aim for simplicity in your sentence structure and vocabulary.
#             - Clarity: Ensure that the sentence is clear and understandable.
#             - Relevance: The sentence should reflect the meaning conveyed by the ASL words.
#             - Grammar: Maintain proper grammar and syntax in the sentence.

#             Example:

#             Input: recognized_words = cat mat
#             output: cat on the mat

#             Here is the actual input for which you have to produce the relevant output: recognised_words = {' '.join(recognised_words)}
#             """
    
#     response = model.generate_content(prompt)
    
#     return response.text

# def load_relevant_data_subset(pq_path):
#     data_columns = ['x', 'y', 'z']
#     data = pd.read_parquet(pq_path, columns=data_columns)
#     ROWS_PER_FRAME = 543
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)

# def do_capture_loop(xyz, pred_fn):
#     all_landmarks = []
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()  # Check if the camera is working and get a frame to read dimensions
#     if not ret:
#         print("Failed to grab frame")
#         cap.release()
#         return
    
#     frame_height, frame_width = frame.shape[:2]
#     scale_factor = 1.0  # Scale the image to fill the window more
#     scaled_height = int(frame_height * scale_factor)
#     scaled_width = int(frame_width * scale_factor)
#     display_width = scaled_width + 1200  # Extra width for text
#     display_height = scaled_height  # Match the height of the camera feed

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 2.5  # Larger font size
#     text_thickness = 3
    
#     start_time = time.time()
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 2.0
#     font_scale_d = 1.1 # Increased font scale for larger text
#     last_prediction_time = 0
#     escape_pressed = False
#     display_message = "Press Escape to toggle message display"
#     unique_signs = []
#     sign_name = ""

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#             current_time = time.time()
#             elapsed_time = int(current_time - start_time)

#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 continue

#             # Scaling up the camera feed
#             image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#             landmarks = create_frame_landmark_df(results, elapsed_time, xyz)
#             all_landmarks.append(landmarks)

#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             mp_drawing.draw_landmarks(
#                 image,
#                 results.face_landmarks,
#                 mp_holistic.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
#             )
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.pose_landmarks,
#                 mp_holistic.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#             )

#             if current_time - last_prediction_time >= 3:
#                 if all_landmarks:
#                     concatenated_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
#                     concatenated_landmarks.to_parquet("out.parquet")
#                     xyz_np = load_relevant_data_subset("out.parquet")
#                     p = pred_fn(inputs=xyz_np)
#                     sign = p['outputs'].argmax()
#                     sign_name = ORD2SIGN[sign]
#                     if sign_name not in unique_signs:
#                         unique_signs.append(sign_name)

#                     last_prediction_time = current_time
#                     all_landmarks = []  # Reset landmarks

#             if sign_name == "" or sign_name == "TV":
#                 sign_name = "No Movement Detected"

#             # UI Improvements
#             display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
#             display[:scaled_height, :scaled_width] = image

#             # Draw the text
#             cv2.putText(display, f"Sign: {sign_name}", (scaled_width + 10, 100), font, font_scale, (0, 255, 0), text_thickness)
#             cv2.putText(display, f"Time: {elapsed_time}s", (scaled_width + 10, 200), font, font_scale, (0, 0, 255), text_thickness)

#             if escape_pressed:
#                 cv2.putText(display, display_message, (scaled_width + 10, 300), font, font_scale_d, (255, 255, 0), text_thickness)

#             cv2.imshow("MediaPipe Holistic", display)


#             key = cv2.waitKey(5)
#             if key & 0xFF == 27:
#                 escape_pressed = not escape_pressed
#                 display_message = get_display_message_from_api(unique_signs) if escape_pressed else "Press Escape to toggle message display"
#                 if escape_pressed: 
#                     unique_signs = []
#             elif key & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# # Load data and start capture loop
# pq_file = "10042041.parquet"
# xyz = pd.read_parquet(pq_file)
# do_capture_loop(xyz, pred_fn)

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
# with mp_holistic.Holistic(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     refine_face_landmarks=True) as holistic:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if results.pose_landmarks:
#       print(
#           f'Nose coordinates: ('
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#       )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose, left and right hands, and face landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.
#         get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()