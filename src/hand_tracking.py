import cv2
import mediapipe as mp
import tensorflow as tf

import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def decode(value):
    class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                     'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
                     'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26}

    key_list = list(class_indices.keys())
    val_list = list(class_indices.values())

    # print key with val 100
    position = val_list.index(value)
    return key_list[position]
# fontScale
fontScale = 1

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

# font
font = cv2.FONT_HERSHEY_SIMPLEX

model = tf.keras.models.load_model('C:/Users/Acer/Desktop/DribhyaDrishti/newmodel/asl_model.h5')


out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    count=0

    while cap.isOpened():
      success, image = cap.read()
      count+=1
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        xList = []
        yList = []
        bbox = []
        lmList = []
        handNo=0
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = image.shape
            px, py = int(lm.x * w), int(lm.y * h)
            xList.append(px)
            yList.append(py)
            lmList.append([px, py])

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH


        roi_x = max(0, bbox[0] - 50)
        roi_y = max(0, bbox[1] - 50)
        roi_h = min(image.shape[0] - roi_y, bbox[3] + 100)
        roi_w = min(image.shape[1] - roi_x, bbox[2] + 100)

        # Draw bounding box
        cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        try:

            resized = cv2.resize(roi, (128,128))
            img_array = np.array([resized])
            # path = 'asl_alphabet_train/asl_alphabet_train/L/x'+str(count)+'.jpg'
            # if count % 5 == 0:
            #   cv2.imwrite(path, resized)
            prediction = model.predict(img_array)

            org = (roi_x + roi_w + 20, roi_y + roi_h+20)

            pred = decode(np.argmax(prediction))
            image = cv2.putText(image, str(pred), org, font, fontScale,
                              color, thickness, cv2.LINE_AA, False)
        except:
          print("nada")

      out.write(image)

      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()