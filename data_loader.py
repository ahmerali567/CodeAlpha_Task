import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data():
    data = np.load("features/features.npz", allow_pickle=True)
    X = np.array(data["features"])
    y = np.array(data["labels"])
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)
    return X, y, le