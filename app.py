import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

model = tf.keras.models.load_model("model.h5")
genre_labels = sorted(os.listdir("spectrograms"))

st.title("🎵 Music Genre Classifier")

uploaded = st.file_uploader("Upload a WAV/MP3 clip", type=["wav", "mp3"])

if uploaded:
    y, sr = librosa.load(uploaded, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(S_DB, sr=sr, ax=ax)
    plt.axis("off")
    st.pyplot(fig)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.resize(img, (64, 64)) / 255.0

    pred = model.predict(np.expand_dims(img, axis=0))[0]
    st.success(f"🎶 Predicted Genre: {genre_labels[np.argmax(pred)]}")
