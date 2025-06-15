import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Acer/Desktop/DribhyaDrishti/newmodel/Dribya_Dristi.h5')

# Define the labels
labels = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    # Show preprocessed ROI for debugging
    cv2.imshow("ROI", resized)
    
    # Predict the label
    prediction = model.predict(reshaped)
    print("Prediction:", prediction)
    label = labels[np.argmax(prediction)]
    #  show prediction precentage confidence
    confidence = np.max(prediction) * 100
    label = f"{label} ({confidence:.2f}%)"

    # Display the label on the frame
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
