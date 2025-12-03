import streamlit as st
from data_loader import download_data
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set page configuration
st.set_page_config(page_title="Stethoscope Disease Detection", layout="wide")
st.title("Digital Stethoscope: Heart and Lung Sound Analysis")

# Download the dataset
dataset_name = "paultimothymooney/chest-xray-pneumonia"  # Specify the dataset name
download_data(dataset_name)

# Function for audio feature extraction (MFCCs)
def extract_mfcc(audio_path):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        st.error(f"Error extracting MFCCs: {e}")
        return None

# UI for uploading an audio file
audio_file = st.file_uploader("Upload a Stethoscope Audio File", type=["wav", "mp3"])

if audio_file is not None:
    # Save uploaded file to local system
    audio_path = f"assets/{audio_file.name}"
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(audio_path)

    # Extract features and display them
    mfcc_features = extract_mfcc(audio_path)
    if mfcc_features is not None:
        st.write("MFCC Features Extracted from Audio:")
        st.write(mfcc_features)

        # Visualize the MFCC features
        fig, ax = plt.subplots()
        ax.plot(mfcc_features)
        ax.set_title("MFCC Feature Plot")
        st.pyplot(fig)

        # Disease detection (dummy model for now)
        # This part should be replaced with your trained model for actual prediction
        st.write("Model Prediction: Cardiovascular or Pulmonary Disease Detected!")
