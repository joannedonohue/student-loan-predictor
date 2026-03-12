# Student Loan Repayment Predictor

A deep neural network that predicts student loan repayment success — built with TensorFlow/Keras to classify borrower outcomes based on financial and academic features.

## Overview

This model helps identify which borrowers are most likely to successfully repay student loans, supporting risk assessment in educational lending. Using a multi-layer neural network trained on borrower profile data, it outputs a binary repayment prediction with 75.5% accuracy.

## Model Architecture

Input layer (11 features) -> Dense(6, relu) -> Dense(3, relu) -> Dense(1, sigmoid)

- Input: 11 borrower features (credit ranking, financial aid history, GPA, etc.)
- Two hidden layers with ReLU activation
- Output layer with sigmoid for binary classification
- Optimizer: Adam | Loss: binary_crossentropy

## Performance

- Test Accuracy: **75.5%**

## Methodology

1. Preprocessed and scaled borrower feature data
2. Split into train/test sets (80/20)
3. Built sequential Keras model; tuned layer sizes and activation functions
4. Evaluated on held-out test set; exported model as .keras file

## Stack

Python | TensorFlow | Keras | pandas | scikit-learn

## Usage

Run: pip install -r requirements.txt
Then: jupyter notebook student_loans_with_deep_learning.ipynb
