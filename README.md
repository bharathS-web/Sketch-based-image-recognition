# Sketch-based Image Recognition Model

This project involves developing a sketch-based image recognition model trained on the Quick Draw dataset. The model aims to classify hand-drawn sketches into various categories, leveraging deep learning techniques to enhance recognition accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

The goal of this project is to build a robust image recognition model capable of recognizing a range of sketches from the Quick Draw dataset. The model learns the features and patterns in hand-drawn sketches, allowing it to generalize well to various sketch categories.

## Dataset

The Quick Draw dataset contains millions of labeled sketches across various categories, drawn by users worldwide. It provides a unique challenge due to its simplicity and variation in drawing styles.

[Quick Draw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false)

## Model Training

The model training pipeline is set up to:

1. Preprocess the Quick Draw dataset for efficient loading and training.
2. Train the sketch recognition model using deep learning techniques.
3. Evaluate model performance and optimize it based on accuracy and loss metrics.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bharathS-web/Sketch-based-image-recognition.git
   cd quick-draw-recognition
   
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Running streamlit file:
   ```bash
   streamlit run quickdraw.py



Usage

1. Data Preparation: Ensure the Quick Draw dataset is available in the data/ directory.


2. Training the Model: Run the model training script in notebooks/ or src/.


3. Evaluation: After training, evaluate the model using the test set and review results.



Results

Once trained, the model should achieve an accuracy that allows it to classify common sketch categories effectively. Further experiments may be conducted to enhance accuracy.

Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to suggest improvements.
