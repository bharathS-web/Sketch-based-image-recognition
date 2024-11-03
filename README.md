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

## CNN Architecture

# model
This repository contains a Convolutional Neural Network (CNN) implementation using Keras. The model is designed for image classification tasks with a multi-layer architecture combining convolutional, pooling, and dense layers.

# Architecture Details
```bash
   model = keras.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

# Layer-by-Layer Description
# Convolutional Blocks

# First Convolutional Block
Conv2D: 16 filters, 3×3 kernel, same padding, ReLU activation
Batch Normalization
MaxPooling: 2×2 pool size

# Second Convolutional Block
Conv2D: 32 filters, 3×3 kernel, same padding, ReLU activation
Batch Normalization
MaxPooling: 2×2 pool size

# Third Convolutional Block
Conv2D: 64 filters, 3×3 kernel, same padding, ReLU activation
Batch Normalization
MaxPooling: 2×2 pool size

# Classification Layers
Flatten: Converts 3D feature maps to 1D feature vectors
Dropout: 50% dropout rate for regularization
Dense: 128 units with ReLU activation
Output: Dense layer with num_classes units and softmax activation

# Architecture Diagram
flowchart TD
    input[Input Layer] --> conv1[Conv2D: 16 filters, 3x3]
    conv1 --> bn1[BatchNormalization]
    bn1 --> pool1[MaxPooling2D: 2x2]
    
    pool1 --> conv2[Conv2D: 32 filters, 3x3]
    conv2 --> bn2[BatchNormalization]
    bn2 --> pool2[MaxPooling2D: 2x2]
    
    pool2 --> conv3[Conv2D: 64 filters, 3x3]
    conv3 --> bn3[BatchNormalization]
    bn3 --> pool3[MaxPooling2D: 2x2]
    
    pool3 --> flatten[Flatten]
    flatten --> dropout[Dropout: 0.5]
    
    dropout --> dense1[Dense: 128 units]
    dense1 --> output[Dense: num_classes units]

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
   ```
   
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Running streamlit file:
   ```bash
   streamlit run quickdraw.py
   ```



Usage

1. Data Preparation: Ensure the Quick Draw dataset is available in the data/ directory.


2. Training the Model: Run the model training script in notebooks/ or src/.


3. Evaluation: After training, evaluate the model using the test set and review results.



Results

Once trained, the model should achieve an accuracy that allows it to classify common sketch categories effectively. Further experiments may be conducted to enhance accuracy.

Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to suggest improvements.
