import os
import numpy as np
import librosa
from tqdm import tqdm

DATA_DIR = "data/RAVDESS"
OUT_FILE = "features/features.npz"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 4  # seconds

def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # (time_steps, 40)
    return mfcc


def get_emotion(filename):
    part = filename.split('-')[2]
    emotions = {
        '01':'neutral','02':'calm','03':'happy','04':'sad',
        '05':'angry','06':'fearful','07':'disgust','08':'surprised'
    }
    return emotions.get(part, 'unknown')

def main():
    features, labels = [], []
    for root, _, files in os.walk(DATA_DIR):
        for f in tqdm(files):
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                mfcc = extract_features(path)
                label = get_emotion(f)
                features.append(mfcc)
                labels.append(label)
    np.savez(OUT_FILE, features=features, labels=labels)
    print("✅ Features saved at", OUT_FILE)

if __name__ == "__main__":
    main()
