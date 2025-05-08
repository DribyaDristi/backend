import io
import matplotlib.pyplot as plt
from tensorflow import keras

# Path to the directory containing the saved model
model_path = 'C:/Users/Acer/Desktop/DribhyaDrishti/newmodel/Dribya_Dristi.h5'

# Load the model
model = keras.models.load_model(model_path)

model.summary()
