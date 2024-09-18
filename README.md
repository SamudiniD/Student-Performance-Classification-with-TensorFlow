# Student-Performance-Classification-with-TensorFlow
## Description
This project aims to classify student performance into three categories: Low, Middle, and High, using a tabular dataset. The project utilizes TensorFlow and Keras to build and train a neural network, followed by hyperparameter tuning with Keras Tuner. It implements early stopping to prevent overfitting and optimizes several hyperparameters such as learning rate, batch size, and dropout rate.

## Dataset
The dataset contains 480 student records with 17 features, including:

Demographic Features: Gender, Nationality, Place of Birth
Academic Features: Educational Stage, Grade Level, Section
Behavioral Features: Raised hand in class, Visited resources, Discussion group participation, etc.

## Requirements
To run this project, you need the following Python packages:

pip install pandas scikit-learn tensorflow keras-tuner

## Model Architecture
The model consists of:

Input Layer: Accepts 17 input features.
Hidden Layers: Two dense layers with 64 and 32 units respectively, ReLU activation.
Dropout Layer: Dropout of 0.3 to prevent overfitting.
Output Layer: 3 units with Softmax activation for multi-class classification.
Training Process
The model is trained using sparse_categorical_crossentropy loss and the Adam optimizer. It was trained for 50 epochs with a batch size of 32. Early stopping was applied to prevent overfitting based on the validation loss.

## Hyperparameter Tuning
Hyperparameter tuning is performed using Keras Tuner, where the following parameters were tuned:

Learning rate: Tuned between 0.0001 and 0.01
Batch size: Tuned between 16, 32, and 64
Number of neurons in the hidden layers
