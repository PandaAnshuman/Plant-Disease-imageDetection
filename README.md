# Plant Disease Recognition System

This project implements a deep learning-based system for recognizing plant diseases using convolutional neural networks (CNNs). The system allows users to upload images of plant leaves and predicts the type of disease present based on the trained model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

The Plant Disease Recognition System is built using TensorFlow and Streamlit. It consists of a trained CNN model capable of classifying 38 different types of plant diseases. Users can interact with the system through an intuitive web interface developed using Streamlit, where they can upload images and receive real-time predictions about the presence of diseases in plant leaves.

## Features

- User-friendly web interface for uploading and analyzing plant images.
- Deep learning model capable of accurately classifying 38 types of plant diseases.
- Interactive visualization of model training history and evaluation metrics.

## Dependencies

- Python 3.x
- TensorFlow
- Streamlit
- Matplotlib
- Pandas
- Seaborn

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/plant-disease-recognition.git
cd plant-disease-recognition
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset and organize the directory structure:

   - Place training images in the `train` directory.
   - Place validation images in the `valid` directory.

## Usage

Run the Streamlit app to launch the web interface:

```bash
streamlit run app.py
```

Access the app in your web browser at `http://localhost:8501`.

## Model Training

1. Adjust hyperparameters and model architecture in `model.py`.
2. Train the model:

```bash
python model.py
```

3. Save the trained model:

```bash
# Save the model in Keras format
model.save("trained_model.keras")
```

## Evaluation

Visualize model training history and evaluate performance using:

```bash
python evaluate_model.py
```

## Contributors

- Your Name ([LinkedIn](Your LinkedIn Profile Link))
- Animesh-py ([GitHub](Animesh-py GitHub Profile Link))

## Acknowledgments

- Animesh-py for contributions and support in the project.
- Dataset source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
