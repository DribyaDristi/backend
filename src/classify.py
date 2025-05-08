import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("models/Dribya_Dristi.h5")
class_labels = label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
# with open("models\class_names.txt", "r") as f:
#     class_labels = f.read().splitlines()


def classify_image(image):
    image_resized = cv2.resize(image, (48, 48))
    
    # Convert to grayscale if model expects single channel
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_gray = np.expand_dims(image_gray, axis=-1)  # (64, 64, 1)

    image_norm = image_gray.astype("float32") / 255.0
    input_data = np.expand_dims(image_norm, axis=0)   # (1, 64, 64, 1)

    prediction = model.predict(input_data)
    label_index = np.argmax(prediction, axis=1)[0]
    return class_labels[label_index]


def classify_static_image_file(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Image could not be loaded. Check file path.")
    return classify_image(image)
