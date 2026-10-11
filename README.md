# 🍃 Plant Pathology - Fine-Grained Image Classification

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/API-Keras-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-green.svg)](https://streamlit.io/)

This repository implements a deep learning computer vision solution for the **Plant Pathology** challenge. The objective is to accurately classify fine-grained disease categories in plant leaves from high-resolution images.

---

## 📊 Project Overview
* **Objective:** Train a robust image classification model to detect healthy leaves, rust, scab, and other complex pathological conditions.
* **Key Steps:**
  * Image preprocessing, resizing, and data augmentation pipeline to handle varying field conditions.
  * Transfer Learning utilizing pre-trained architectures (such as EfficientNet or ResNet) fine-tuned on plant pathology datasets.
  * Model evaluation using multi-class classification metrics (ROC-AUC, Precision, Recall, and F1-score).
  * Interactive web dashboard deployment via **Streamlit** for real-time leaf disease diagnosis.

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **TensorFlow & Keras** (Deep Learning & Transfer Learning)
* **OpenCV & PIL** (Image Processing & Augmentation)
* **NumPy & Pandas** (Data Management)
* **Matplotlib & Seaborn** (Performance Visualization)
* **Streamlit** (Interactive Dashboard)

---

## 🚀 Getting Started & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tomris17/plant-pathology-image-classification.git](https://github.com/tomris17/plant-pathology-image-classification.git)
   cd plant-pathology-image-classification
