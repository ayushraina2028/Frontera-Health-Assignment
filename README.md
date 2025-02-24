# Audio Classification Web Application

## Overview
This project is a Streamlit-based web application that classifies audio files using an ensemble of two deep learning models: **YAMNet** and **Wav2Vec2**. The app takes a WAV audio file as input and provides class predictions using different ensemble techniques.

## Report
A small presentation including all the experiments and results involved in this project is available in the Report Folder. It can be directly accessed using this Link: [Report](https://github.com/ayushraina2028/Frontera-Health-Assignment/blob/master/Report/report.pdf) 

## Features
- Upload a WAV audio file
- Get predictions from **YAMNet** and **Wav2Vec2** models
- Apply ensemble techniques:
  - **Majority Voting**
  - **Weighted Voting**
- Display predictions and probabilities in a well-structured UI

## Downloading Weights
The weights for YAMNet model are available in the above files with the name as "yamnet_finetuned.h5". The weights for Wav2Vec2 model are large in size and hence cannot be uploaded to the repository. Please download weights from this [link](https://drive.google.com/file/d/1wIGGLKuEKHiK3rPd6ge_PvtAUoIMTOlM/view?usp=sharing) to run the web application properly.

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

## Usage of Web Application
You can upload a WAV audio file using the file uploader in the web application. The app will then display the predictions from the two models and the ensemble predictions using different techniques.

To run the trained Yamnet Model, open this [File](https://github.com/ayushraina2028/Frontera-Health-Assignment/blob/master/Ensemble1.ipynb)

To run the trained Wav2Vec2 Model, open this [File](https://github.com/ayushraina2028/Frontera-Health-Assignment/blob/master/Ensemble2.ipynb) Given that you have downloaded weights from the link provided above.