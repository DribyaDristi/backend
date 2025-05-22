from tensorflow import keras
from keras.layers import TFSMLayer
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv
import os

def LoadModel():
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'Emotion_Recognizer')
    input_shape = (48, 48, 1) 
    inputs = keras.Input(shape=input_shape)
    
    outputs = TFSMLayer(model_path, call_endpoint='serving_default')(inputs)  
    custom_model = keras.Model(inputs, outputs)
    return custom_model

def preprocess_image(face_img):
    img = cv.resize(face_img, (48, 48))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)  # Apply histogram equalization
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    img = img / 255.0
    return img

def predict_emotion(face_img):
    custom_model=LoadModel()
    labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    img = preprocess_image(face_img)
    pred = custom_model.predict(img)
    pred_array = pred['dense_5']
    pred_index = np.argmax(pred_array)
    pred_label = labels[pred_index]
    return pred_label



"""
# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face_img)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the emotion label
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Print emotion label in terminal
        print("Detected emotion:", emotion)
    
    # Display the frame with rectangles and emotion labels
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

def Recognition(image_path):
    face_img = cv2.imread(image_path)
    emotion = predict_emotion(face_img)
    return emotion
"""

def recognize_emotion(image_path):
    global recognized_emotion
    recognized_emotion = []


    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_file_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')

    img = cv.imread(image_path)

    face_cascade = cv.CascadeClassifier(cascade_file_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        emotion = predict_emotion(face_img)
        recognized_emotion.append({"emotion": emotion, "coordinates": {"x": x, "y": y, "w": w, "h": h}})
        print(f"Detected emotion: {emotion} at coordinates: x={x}, y={y}, w={w}, h={h}")
        
    return recognized_emotion

name = recognize_emotion("C:\\Users\\Swarnim Bajracharya\\Downloads\\Suhsil A\\IMG_3590.jpg")

