import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import json

# Load model and class labels
model = tf.keras.models.load_model("isl_cnn_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = [label for label, idx in sorted(class_indices.items(), key=lambda item: item[1])]

# Video transformer
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (64, 64))
        img_input = np.expand_dims(img_resized / 255.0, axis=0)

        prediction = model.predict(img_input)
        pred_index = np.argmax(prediction)
        label = class_labels[pred_index]
        confidence = np.max(prediction)

        cv2.putText(img, f"{label} ({confidence*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return img

# Streamlit UI
st.title("Indian Sign Language Interpreter")
st.write("Show a hand sign to your webcam. Model will predict the sign.")
webrtc_streamer(key="sign", video_transformer_factory=SignLanguageTransformer)
