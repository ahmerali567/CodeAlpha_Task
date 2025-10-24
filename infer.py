import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib

model = load_model("models/emotion_model.h5")

def predict_emotion(file):
    y, sr = librosa.load(file, sr=16000)
    y = librosa.util.fix_length(y, size=16000*4)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    mfcc = np.expand_dims(mfcc, axis=0)
    pred = model.predict(mfcc)
    emotion = np.argmax(pred)
    print("Predicted Emotion:", emotion)

predict_emotion("data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav")
