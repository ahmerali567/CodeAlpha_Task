from tensorflow.keras.models import load_model
import numpy as np
import librosa

# ✅ Load trained model
model = load_model("models/emotion_model.h5")

# ✅ Test audio path
audio_path = "test_audio1.wav"

# 🎵 Extract MFCC features (same as Flask)
y, sr = librosa.load(audio_path, duration=3, offset=0.5)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc = mfcc.T  # shape: (time_steps, 40)
print("📊 Original shape:", mfcc.shape)

# Pad or trim to match expected shape (126, 40)
expected_shape = (126, 40)
if mfcc.shape[0] < expected_shape[0]:
    pad_width = expected_shape[0] - mfcc.shape[0]
    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
elif mfcc.shape[0] > expected_shape[0]:
    mfcc = mfcc[:expected_shape[0], :]

# Expand dims to make shape (1, 126, 40)
mfcc = np.expand_dims(mfcc, axis=0)
print("📊 Final shape:", mfcc.shape)

# 🎯 Predict emotion
prediction = model.predict(mfcc)
predicted_class = np.argmax(prediction, axis=1)[0]

class_labels = ['angry', 'happy', 'neutral', 'sad', 'fear', 'disgust', 'surprise']
emotion = class_labels[predicted_class] if predicted_class < len(class_labels) else "unknown"

print(f"🎯 Predicted Emotion: {emotion}")

