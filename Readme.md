# 🎭 Emotion Detection in Text using LSTMs

## 📘 Project Overview

This project develops a deep learning model to detect **emotions from text**, especially from social media-style messages (e.g., tweets, comments). The model classifies text into one of six categories:

> **joy**, **sadness**, **anger**, **fear**, **love**, and **surprise**

The goal is to enable analysis of public sentiment, identify emotional cues in crisis communication, and assess the emotional impact of global events.

---

## 👥 Group Members

* **Jenil Kevadiya** (Matriculation No.: 22204227)

* **Om Vaghasiya** (Matriculation No.: 22205283)

* **Krunal Koladiya** (Matriculation No.: 22306168)

* **Venkat Rajasekar** (Matriculation No.: 22308591)

---

## 🚀 Features

* ✅ **Bidirectional LSTM Model** for emotion classification

* 🔤 **GloVe Word Embeddings** for semantic understanding

* 🧹 **Preprocessing Pipeline**: cleaning, stopword removal, lemmatization

* 🔁 **Stratified K-Fold Cross-Validation** (in `main.py`)

* 🖥️ **Tkinter GUI (Optional)** for real-time predictions (in `main.py` and adaptable to the notebook)

---

## 📂 Dataset

* 📥 **Source:** The model is trained on the "Emotions Dataset for NLP" available on Kaggle:
  [Emotions Dataset for NLP (Kaggle)](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

### Expected Directory Structure:

The dataset and embeddings are expected to be in the following directory structure relative to your script/notebook:

```
Dataset/
├── train.txt
├── val.txt
└── test.txt

Glove/
└── glove.6B.200d.txt

```

* 🔗 **GloVe Embeddings (200d):** Download from Kaggle:
  [GloVe 6B 200d Embeddings](https://www.kaggle.com/datasets/incorpes/glove6b200d)

*Ensure* that the dataset and embeddings are placed in the correct directories. You may need to update paths in the scripts *if your structure differs.*

---

## 🛠️ Setup & Installation

### ✅ Prerequisites

* Python 3.x

* pip (Python package installer)

### 📦 Dependencies

Install the required Python packages using the `requirements.txt` file:

```
pip install -r requirements.txt

```

###  System Dependencies

**Important** Note on **Graphviz (for model plotting):**

The `pydot` library, used for visualizing the model architecture via Keras' `plot_model` function, requires **Graphviz** to be installed on your system. This is a system-level dependency, not just a Python package.

* **Installation Instructions:** Please refer to the official Graphviz download page:
  <https://graphviz.gitlab.io/download/>

* **PATH Configuration:** After installing Graphviz, ensure that its `bin` directory is added to your system's PATH environment variable. This allows your system to find the Graphviz executables (like `dot`).

* **Troubleshooting:** If `plot_model` in Keras still fails after installation and PATH configuration (e.g., with a "dot: command not found" error), you might need to:

  * Restart your machine.

  * Restart your development environment (e.g., Jupyter kernel, IDE).

  * Verify the PATH variable was updated correctly.

###  NLTK Resources

The notebook and script will attempt to download necessary NLTK resources (stopwords, wordnet, omw-1.4, punkt) if they are not found. Please ensure you have an active internet connection during the first run to allow these downloads. If automatic download fails, you may need to open a Python interpreter and manually download them:

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

```

---

##  How to Run

### Jupyter Notebook (`Emotion_with_lstm.ipynb`)

1. **Dependencies & Setup:**

   * Ensure all Python dependencies listed in `requirements.txt` are installed.

   * Confirm Graphviz is installed and configured in your system's PATH if you wish to generate model architecture plots.

   * Verify NLTK resources are downloaded.

2. **Data Placement:**

   * Place the dataset files (`train.txt`, `val.txt`, `test.txt`) in a subdirectory named `Dataset`.

   * Place the GloVe embeddings file (`glove.6B.200d.txt`) in a subdirectory named `Glove`.

3. **Launch Notebook:** Open `Emotion_with_lstm.ipynb` using Jupyter Lab or Jupyter Notebook.

4. **Execute Cells:** Run the cells sequentially from top to bottom for the complete workflow, including data loading, preprocessing, model training, evaluation, and saving artifacts.

*(Note:* The `main.py` script, if used for its K-Fold cross-validation or GUI, requires the same *setup steps as the Jupyter Notebook regarding dependencies, dataset, and GloVe files.)*

---

##  Project Structure & Key Files

* **`Emotion_with_lstm.ipynb`**: The primary Jupyter Notebook detailing data exploration, preprocessing, model training (single split), and evaluation.

* **`main.py` (Optional):** A Python script offering a more structured workflow with K-Fold cross-validation and an interactive Tkinter GUI.

* **`requirements.txt`**: Lists Python package dependencies.

* **`README.md`**: This file.

* **`Dataset/`**: Directory for `train.txt`, `val.txt`, `test.txt`.

* **`Glove/`**: Directory for `glove.6B.200d.txt`.

* **`Emotion_Recognition_Final_Model.keras`**: The trained Keras model (filename may vary slightly based on which script is used for saving).

* **`tokenizer.json`**: Saved Keras tokenizer.

* **`label_encoder.npy`**: Saved label encoder classes.

* **Model Card (PDF/LaTeX):** A separate document providing a comprehensive overview of the model.

---

##  Model Details (Brief)

* **Architecture:** Bidirectional LSTM network utilizing pre-trained GloVe word embeddings.

* **Optimizer:** Adam.

* **Loss Function:** Categorical Cross-Entropy.

*For comprehensive details on the model, its training procedure, and in-depth evaluation, please refer to the accompanying **Model Card** document.*

---

##  Limitations & Ethical Considerations

* **Language Bias:** The model may inherit biases from the training data and the pre-trained GloVe embeddings.

* **Context & Sarcasm:** The model does not deeply understand complex linguistic nuances such as context, irony, or sarcasm.

* **Dataset Representativeness:** The dataset is primarily sourced from social media and may not generalize perfectly to other forms of text or all emotional expressions.

* **Fixed Emotion Set:** The model is limited to classifying text into the six predefined emotions (joy, sadness, anger, fear, love, surprise).

*A* full *discussion of these points and other considerations can be found in the **Model Card** document.*
