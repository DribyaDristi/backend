import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tflite_runtime.interpreter as tftite
# from tensorflow.lite import Interpreter
import tensorflow.lite as tflite

ROWS_PER_FRAME = 543 # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = data.shape[0] // ROWS_PER_FRAME
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

interpreter = tflite.Interpreter("./model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('C:/Users/Acer/Desktop/DribhyaDrishti/src/word/train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes

SIGN2ORD = train[['sign','sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord','sign']].set_index('sign_ord').squeeze().to_dict()

pq_file = 'C:/Users/Acer/Desktop/DribhyaDrishti/src/word/output.parquet'
xyz_np = load_relevant_data_subset(pq_file)
prediction = prediction_fn(inputs=xyz_np)
sign = prediction['outputs'].argmax()

print(f"Predicted sign: {ORD2SIGN[sign]} (ord: {sign})")