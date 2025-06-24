import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import sys

model = tf.keras.models.load_model("model.h5")
genre_labels = list(os.listdir("spectrograms/"))

def preprocess_audio(file):
    y, sr = librosa.load(file, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24))
    librosa.display.specshow(S_DB, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    img = cv2.resize(img, (64, 64)) / 255.0
    return np.expand_dims(img, axis=0)

file = sys.argv[1]
input_img = preprocess_audio(file)
pred = model.predict(input_img)[0]
print("🎵 Predicted Genre:", genre_labels[np.argmax(pred)])
