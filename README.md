# Audio Classification Web Application

## Overview
This project is a Streamlit-based web application that classifies audio files using an ensemble of two deep learning models: **YAMNet** and **Wav2Vec2**. The app takes a WAV audio file as input and provides class predictions using different ensemble techniques.

## Features
- Upload a WAV audio file
- Get predictions from **YAMNet** and **Wav2Vec2** models
- Apply ensemble techniques:
  - **Majority Voting**
  - **Weighted Voting**
- Display predictions and probabilities in a well-structured UI

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. Install the required packages:
   ```bash
    pip install -r requirements.txt

3. To start the Streamlit Web Application:
   ```bash
   streamlit run ensemble.py

## Downloading Datasets
To run the jupiter notebook files in this repository, you need to download the datasets used in the models.

First download the dataset of baby-cry sounds from [Kaggle](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus?utm_source=chatgpt.com). Extract the contents of the ZIP file and place all the WAV files inside Datasets/Cry folder.

Then download this dataset again from [Kaggle](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset?utm_source=chatgpt.com&select=NotScreaming). Extract the two folders and place them inside Datasets/ folder.

The final structure of the Datasets folder should look like this:
```
Datasets/
├── Cry/
│   ├── 1-100032-A-0.wav
│   ├── 1-100038-A-14.wav
│   ├── ...
├── NotScreaming/
│   ├── 1.wav
│   ├── 2.wav
│   ├── ...
├── Screaming/
│   ├── 1.wav
│   ├── 2.wav
│   ├── ...
```
```
## Usage of Web Application
You can upload a WAV audio file using the file uploader in the web application. The app will then display the predictions from the two models and the ensemble predictions using different techniques.