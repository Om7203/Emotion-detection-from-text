# 🎭 Emotion Detection in Text using LSTMs

## 📘 Project Overview

This project develops a deep learning model to detect **emotions from text**, especially from social media-style messages (e.g., tweets, comments). The model classifies text into one of six categories:

> **joy**, **sadness**, **anger**, **fear**, **love**, and **surprise**

The goal is to enable real-time analysis of public sentiment, identify emotional cues in crisis communication, and assess the emotional impact of global events.

---

## 👥 Group Members

- **Jenil Kevadiya** (Matriculation No.: 22204227)  
- **Om Vaghasiya** (Matriculation No.: 22205283)  
- **Krunal Koladiya** (Matriculation No.: 22306168)  
- **Venkat Rajasekar** (Matriculation No.: 22308591)

---

## 🚀 Features

- ✅ **Bidirectional LSTM Model** for emotion classification  
- 🔤 **GloVe Word Embeddings** for semantic understanding  
- 🧹 **Preprocessing Pipeline**: cleaning, stopword removal, lemmatization  
- 🔁 **Stratified K-Fold Cross-Validation**  
- 🖥️ **Tkinter GUI (Optional)** for real-time predictions (`main.py`)

---

## 📂 Dataset

- 📥 Download the dataset from Kaggle:  
  [Emotions Dataset for NLP (Kaggle)](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

### Expected Directory Structure:

The dataset is expected to be in a directory structure like:

Dataset/
├── train.txt
├── val.txt
└── test.txt

Glove/
└── glove.6B.200d.txt

(
> 🔗 [Download GloVe Embeddings (200d)](https://www.kaggle.com/datasets/incorpes/glove6b200d)

Ensure that the dataset and embeddings are placed in the correct directories and update paths in scripts if needed.

---

## 🛠️ Setup & Installation

### ✅ Prerequisites
- Python 3.x
- pip

### 📦 Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt


###⚙️ Prerequisites and Setup

System Dependencies
Important Note on Graphviz (for model plotting):

The pydot library, used for visualizing the model architecture, requires Graphviz to be installed on your system. This is a system dependency, not just a Python package.

Installation Instructions: Please refer to the official Graphviz download page:
https://graphviz.gitlab.io/download/

PATH Configuration: After installing Graphviz, ensure that its bin directory is added to your system's PATH environment variable.

Troubleshooting: If plot_model in Keras still fails after installation and PATH configuration, you might need to restart your machine or your development environment (e.g., Jupyter kernel, IDE).

NLTK Resources
The notebook and script will attempt to download necessary NLTK resources (stopwords, wordnet, omw-1.4, punkt) if they are not found. Please ensure you have an active internet connection during the first run to allow these downloads.

How to Run
Jupyter Notebook (Emotion_with_lstm.ipynb)
Dependencies: Ensure all Python dependencies listed in requirements.txt are installed.

Graphviz: Confirm Graphviz is installed on your system and configured in your PATH if you wish to generate model architecture plots.

Dataset:

Place the dataset files (train.txt, val.txt, test.txt) in a subdirectory named Dataset within your project folder.

GloVe Embeddings:

Place the GloVe embeddings file (glove.6B.200d.txt) in a subdirectory named Glove within your project folder.

Launch Notebook: Open Emotion_with_lstm.ipynb using Jupyter Lab or Jupyter Notebook.

Execute Cells: Run the cells sequentially from top to bottom.

(Note: The main.py script, if used, requires the same setup steps as the Jupyter Notebook for dependencies, dataset, and GloVe files.)

Project Structure & Key Files
Emotion_with_lstm.ipynb: The primary Jupyter Notebook containing detailed steps for data exploration, preprocessing, model training, and evaluation.

Emotion_Recognition_Final_Model.keras: The trained Keras model will be saved with this filename after successful training.

tokenizer.json: The Keras tokenizer, configured during preprocessing, will be saved in this JSON file.

label_encoder.npy: The label encoder classes (mapping emotions to numerical values) will be saved in this NumPy file.

Model Details (Brief)
Architecture: Bidirectional LSTM network utilizing pre-trained GloVe word embeddings.

Optimizer: Adam.

Loss Function: Categorical Cross-Entropy.

For comprehensive details on the model, its training procedure, and in-depth evaluation, please refer to the accompanying Model Card document.

Limitations & Ethical Considerations
Language Bias: The model may inherit biases from the training data and the pre-trained GloVe embeddings.

Context & Sarcasm: The model does not deeply understand complex linguistic nuances such as context, irony, or sarcasm.

Dataset Representativeness: The dataset is primarily sourced from social media and may not generalize perfectly to other forms of text or all emotional expressions.

Fixed Emotion Set: The model is limited to classifying text into the six predefined emotions (joy, sadness, anger, fear, love, surprise).

A full discussion of these points and other considerations can be found in the Model Card document.
