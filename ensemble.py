import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', category=UserWarning)  # Suppress specific warnings
import torch
torch.set_grad_enabled(False)
import torchaudio
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_hub as hub
import librosa
import pickle
import sys

import streamlit as st

# Load class mapping
CLASS_MAPPING = {0: "Cry", 1: "Screaming", 2: "NotScreaming"}

# ✅ Force TensorFlow to run on CPU (Prevents GPU memory usage)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)
    waveform = waveform.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    tf.keras.backend.clear_session()  # ✅ Free memory
    return np.mean(embeddings.numpy(), axis=0)

def get_yamnet_prediction(model,filename):
    embedding = extract_embedding(filename)
    embedding = np.expand_dims(embedding, axis=0)
    prediction = model.predict(embedding)
    predicted_class = np.argmax(prediction)
    
    # print(f"Predicted Class: {CLASS_MAPPING[predicted_class]}")
    # print(f"Prediction Scores: {prediction[0]}")
    
    answer = CLASS_MAPPING[predicted_class]
    probabilities = prediction[0]
    return answer, probabilities

# Set GPU device to visible again
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(Wav2VecClassifier, self).__init__()
        
        # Load the pre-trained Wav2Vec model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze the feature extractor layers (optional)
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
            
        # Get the output dimension of wav2vec
        hidden_size = self.wav2vec.config.hidden_size  # typically 768
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Get wav2vec features
        outputs = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state
        
        # Pool the output (take mean of all time steps)
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits

# Let's test the model with random data
def test_model():
    # Create model instance
    model = Wav2VecClassifier(num_classes=3)
    model.eval()
    
    # Create random input tensor
    # Wav2Vec expects input shape: [batch_size, sequence_length]
    # Typical audio sampling rate is 16kHz, let's create 1 second of audio
    batch_size = 2
    sequence_length = 16000  # 1 second of audio at 16kHz
    random_audio = torch.randn(batch_size, sequence_length)
    
    # Forward pass
    with torch.no_grad():
        output = model(random_audio)
    
    # print("Input shape:", random_audio.shape)
    # print("Output shape:", output.shape)
    # print("Output (logits):", output)

def preprocess_audio(audio_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to numpy array and squeeze
    waveform = waveform.squeeze().numpy()
    
    # Normalize audio
    waveform = waveform / np.max(np.abs(waveform))
    
    return waveform

def test_model_on_audio(model, audio_path, device):
    class_mapping = {
            'Cry': 0,
            'NotScreaming': 1,
            'Screaming': 2
        }
    
    # Preprocess audio
    waveform = preprocess_audio(audio_path)
    input_values = torch.FloatTensor(waveform).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_values)
    
    # Get predicted class
    _, predicted = torch.max(outputs.data, 1)
    
    # Get class name
    class_mapping = {v: k for k, v in class_mapping.items()}
    predicted_class = class_mapping[predicted.item()]
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    return predicted_class, probabilities

# Load Yamnet model
model_yamnet = tf.keras.models.load_model("yamnet_finetuned.h5")

# Load Wav2Vec model
model_wav2vec = Wav2VecClassifier(num_classes=3)
model_wav2vec.load_state_dict(torch.load('wav2vec_model.pth'))
model_wav2vec.eval()

# Test the model on test set in CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wav2vec = model_wav2vec.to(device)

from collections import Counter

def ensemble_majority_voting(predictions):
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]  # Return the class with the highest count

def ensemble_average_probabilities(prob1, prob2):
    return (np.array(prob1) + np.array(prob2)) / 2

def ensemble_weighted_voting(prob1, prob2):
    avg_probs = ensemble_average_probabilities(prob1, prob2)
    return np.argmax(avg_probs)

st.set_page_config(page_title="Audio Classification", layout="centered")
st.title("🔊 Audio Classification Web App")
st.write("Upload a WAV audio file and get class predictions using an ensemble method.")

st.markdown("---")

audio_file = st.file_uploader("🎵 Upload your audio file", type=["wav"], help="Supports only .wav format")

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    filename = "audio.wav"
    with open(filename, "wb") as f:
        f.write(audio_file.read())
    
    ans_yamnet, probabilities_yamnet = get_yamnet_prediction(model_yamnet, filename)
    ans_wav2vec, probabilities_wav2vec = test_model_on_audio(model_wav2vec, filename, device)
    
    # Convert probabilities to decimal format
    probabilities_yamnet = np.round(probabilities_yamnet, 4).tolist()
    probabilities_wav2vec = np.round(probabilities_wav2vec, 4).tolist()
    
    # Ensemble predictions
    ensemble_class_majority = ensemble_majority_voting([ans_yamnet, ans_wav2vec])
    ensemble_class_weighted = ensemble_weighted_voting(probabilities_yamnet, probabilities_wav2vec)
    
    st.markdown("### 🏆 Model Predictions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("YAMNet")
        st.write(f"**Prediction:** {ans_yamnet}")
        st.write(f"**Scores:** {probabilities_yamnet}")
    
    with col2:
        st.subheader("Wav2Vec")
        st.write(f"**Prediction:** {ans_wav2vec}")
        st.write(f"**Scores:** {probabilities_wav2vec}")
    
    st.markdown("---")
    st.subheader("🎯 Ensemble Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Majority Voting")
        st.write(f"**Final Prediction:** {ensemble_class_majority}")
    
    with col2:
        st.subheader("Weighted Voting")
        st.write(f"**Final Prediction:** {CLASS_MAPPING[ensemble_class_weighted]}")
    
    st.success("✅ Classification completed successfully!")
