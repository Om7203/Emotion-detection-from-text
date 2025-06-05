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

Important Note on Graphviz (for model plotting):
The pydot library, used for visualizing the model architecture, requires Graphviz to be installed on your system. This is a system dependency, not just a Python package.

Installation Instructions for Graphviz: Please refer to the official Graphviz download page: https://graphviz.gitlab.io/download/

After installing Graphviz, ensure that its bin directory is added to your system's PATH environment variable. If plot_model in Keras still fails, you might need to restart your machine or your development environment.

NLTK Resources
The notebook and script will attempt to download necessary NLTK resources (stopwords, wordnet, omw-1.4, punkt) if they are not found. Ensure you have an internet connection during the first run.

How to Run
Jupyter Notebook (Emotion_with_lstm.ipynb)
Ensure all dependencies from requirements.txt are installed and Graphviz is set up if you want model plots.

Place the dataset files (train.txt, val.txt, test.txt) in a subdirectory named Dataset.

Place the GloVe embeddings file (glove.6B.200d.txt) in a subdirectory named Glove.

Open the notebook using Jupyter Lab or Jupyter Notebook.

Run the cells sequentially from top to bottom.

Follow the same setup steps as for the Jupyter Notebook (dependencies, dataset, GloVe).

Project Structure
Emotion_with_lstm.ipynb: Jupyter Notebook with detailed steps for data exploration, preprocessing, model training, and evaluation.

Emotion_Recognition_Final_Model.keras: The trained model will be saved as this given name.

tokenizer.json: Keras tokenizer will be saved as.

label_encoder.npy: label encoder classes will be saved as.

Model Details (Brief)
Architecture: Bidirectional LSTM with GloVe embeddings.

Optimizer: Adam.

Loss Function: Categorical Cross-Entropy.

Refer to the Model Card document for comprehensive details on the model, training, and evaluation.

Limitations & Ethical Considerations
Language Bias: The model may inherit biases from the training data and GloVe embeddings.

Context & Sarcasm: The model does not deeply understand context, irony, or sarcasm.

Dataset Representativeness: The dataset is primarily from social media and may not generalize perfectly to other text forms.

Fixed Emotion Set: Limited to the six predefined emotions.

Please see the Model Card for a full discussion.

