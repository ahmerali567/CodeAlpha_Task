from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
from pydub import AudioSegment
from keras.models import load_model
import tensorflow as tf
import subprocess

# ---------------------------------------------------
# 🔧 Flask setup
# ---------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------
# 🧠 Load your trained model
# ---------------------------------------------------
MODEL_PATH = "models/emotion_model.h5"
  # <-- make sure correct filename
model = load_model(MODEL_PATH)

# ---------------------------------------------------
# 🎵 Feature Extraction
# ---------------------------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # shape: (time_steps, 40)
    return mfcc

# ---------------------------------------------------
# 🎧 Home Route
# ---------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------------------------------
# 🎤 Prediction Route
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Save uploaded audio
        audio_file = request.files['audio']
        file_path = os.path.join(UPLOAD_FOLDER, "recording.wav")
        audio_file.save(file_path)
        print(f"🎤 Audio received: {audio_file.filename}")
        print(f"✅ Saved at: {file_path}")

        # Convert audio to WAV (using pydub + ffmpeg)
        print("🔄 Converting to WAV...")
        try:
            sound = AudioSegment.from_file(file_path)
            wav_path = os.path.splitext(file_path)[0] + ".wav"
            sound.export(wav_path, format="wav")
            file_path = wav_path
            print(f"✅ Converted to: {file_path}")
        except Exception as e:
            print(f"❌ FFmpeg conversion error: {e}")
            return jsonify({'error': 'Audio conversion failed'}), 500

        # Extract features
        features = extract_features(file_path)
        print(f"📊 Original features shape: {features.shape}")

        # Fix shape to (1, 126, 40)
        expected_shape = (126, 40)
        if features.shape[0] < expected_shape[0]:
            pad_width = expected_shape[0] - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        elif features.shape[0] > expected_shape[0]:
            features = features[:expected_shape[0], :]

        features = np.expand_dims(features, axis=0)
        print(f"📊 Final features shape: {features.shape}")

        # Predict emotion
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Class labels (change according to your training)
        class_labels = ['angry', 'happy', 'neutral', 'sad', 'fear', 'disgust', 'surprise']
        emotion = class_labels[predicted_class] if predicted_class < len(class_labels) else "unknown"

        print(f"🎯 Emotion detected: {emotion}")
        return jsonify({'emotion': emotion})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------
# 🚀 Run Flask (for both laptop & mobile access)
# ---------------------------------------------------
if __name__ == "__main__":
    import socket, subprocess

    # Check ffmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ FFmpeg is installed and working.")
    except Exception:
        print("⚠️ FFmpeg not found. Please install it and add to PATH.")
        print("Download: https://ffmpeg.org/download.html")

    # Get local IP for mobile access
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n🌍 Access on laptop: http://127.0.0.1:5000")
    print(f"📱 Access on mobile:  http://{local_ip}:5000\n")

    # Run Flask server on all network interfaces
    app.run(host="0.0.0.0", port=5000, debug=True)


