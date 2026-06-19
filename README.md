# Baby Cry Classification Using Structure-Tuned ANN, MFCC and Data Augmentation

A complete reproduction and analysis of the paper:

**"Baby Cry Classification Using Structure-Tuned Artificial Neural Networks with Data Augmentation and MFCC Features"**

This repository contains the original paper, Persian translation, mathematical formulation extraction, DSP-based analysis, and Python implementations used to reproduce and analyze the reported results.

---

## Project Goals

The main objective of this project is to study, reproduce, and analyze a baby cry classification system based on Digital Signal Processing (DSP) and Artificial Neural Networks (ANNs).

The project includes:

* Original research paper
* Persian translation of the paper
* Mathematical equation extraction
* MFCC feature extraction analysis
* Audio preprocessing and augmentation
* ANN implementation in Python
* Reproduction of experimental results
* Signal-domain and feature-domain visualizations
* DSP interpretation of the proposed method

---

## Paper Overview

The original study proposes a baby cry classification framework consisting of four major stages:

1. Data Augmentation
2. MFCC Feature Extraction
3. Hyperparameter Optimization (Grid Search)
4. Structure-Tuned Artificial Neural Network

The model classifies infant cries into five categories:

* Hungry
* Belly Pain
* Burping
* Discomfort
* Tired

The reported test accuracy reaches approximately **90%** on the Donate-a-Cry dataset.

---

## Digital Signal Processing Analysis

This repository focuses on the DSP aspects of the paper, including:

### Audio Preprocessing

* Resampling
* Padding
* Trimming
* Normalization

### Data Augmentation

* Time Shift
* Time Stretch
* Pitch Shift
* White Noise Injection
* Random Slicing

### Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCC)

The following processing pipeline is implemented and analyzed:

```text
Audio Signal
     │
     ▼
Pre-Emphasis
     │
     ▼
Frame Blocking
     │
     ▼
Windowing
     │
     ▼
FFT
     │
     ▼
Mel Filter Bank
     │
     ▼
Log Energy
     │
     ▼
DCT
     │
     ▼
MFCC Features
```

## Implemented Equations

The repository extracts and implements the main equations used in the paper, including:

* Pre-emphasis filter
* FFT transformation
* Mel-frequency mapping
* DCT-based MFCC computation
* ReLU activation function
* LogSoftmax output function

All equations are accompanied by Python implementations and graphical analysis.

---

## Reproduced Experiments

Implemented experiments include:

* MFCC visualization
* Feature distribution analysis
* ANN training
* Hyperparameter optimization
* Confusion matrix generation
* ROC analysis
* Accuracy and F1-score evaluation

---

## Dataset

Experiments are based on the Donate-a-Cry dataset.

Repository:

https://github.com/gveres/donateacry-corpus

Classes:

* Hungry
* Belly Pain
* Burping
* Discomfort
* Tired

Special thanks to **Gaber Veres** for making the dataset publicly available.

---

## Technologies

* Python
* NumPy
* SciPy
* Librosa
* Matplotlib
* Scikit-Learn
* TensorFlow
* Keras
* Jupyter Notebook

---

## Educational Purpose

This repository is intended for:

* DSP students
* Machine Learning researchers
* Audio Signal Processing enthusiasts
* Researchers interested in cry signal classification

---

## Citation

If you use this repository in academic work, please cite the original paper and the Donate-a-Cry dataset.
