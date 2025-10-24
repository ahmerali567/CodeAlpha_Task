from data_loader import load_data
from models import build_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

X, y, le = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]

model = build_model(input_shape, num_classes)
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)
model.save("models/emotion_model.h5")
print("✅ Model saved at models/emotion_model.h5")
