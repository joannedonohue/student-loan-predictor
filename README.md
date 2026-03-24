# Student Loan Repayment Predictor

A deep neural network that predicts student loan repayment success — built with TensorFlow/Keras to classify borrower outcomes based on financial and academic features.

---

## Overview

This project applies binary classification using a feedforward neural network to predict whether a student borrower will successfully repay their loan. The model is trained on 11 borrower features including credit history, financial aid metrics, and academic standing, achieving 75.5% accuracy on held-out test data.

---

## Model Architecture

```
Input Layer  →  11 features
Dense(6, relu)
Dense(3, relu)
Dense(1, sigmoid)  →  Binary output (repay / default)
```

- **Activation:** ReLU in hidden layers; Sigmoid for binary output
- **Optimizer:** Adam
- **Loss function:** Binary cross-entropy
- **Train/test split:** 80% / 20%

---

## Input Features

The model uses 11 borrower characteristics:
- Credit ranking
- Financial aid history
- GPA
- Additional financial and academic indicators

---

## Performance

| Metric | Value |
|---|---|
| Test accuracy | **75.5%** |
| Output type | Binary classification |

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep learning | TensorFlow / Keras |
| Data preprocessing | scikit-learn, pandas |
| Model export | `.keras` file format |
| Language | Python |

---

## Methodology

1. Load and explore the student loan dataset
2. Preprocess features: scale numerical inputs, encode categoricals
3. Split into 80/20 train/test sets
4. Build a sequential Keras model with two hidden layers
5. Train with Adam optimizer and binary cross-entropy loss
6. Evaluate on test set; export trained model as `.keras` file

---

## Repository Structure

```
student-loan-predictor/
├── student_loans_with_deep_learning.ipynb   # Model training & evaluation
└── README.md
```

---

## Outcomes

- Achieved **75.5% test accuracy** on student loan repayment classification
- Demonstrated a compact, 3-layer neural network architecture suitable for structured financial data
- Exported trained model for potential deployment or further fine-tuning
- Established a reproducible pipeline from raw borrower data through training to evaluation

---

## Getting Started

```bash
pip install tensorflow scikit-learn pandas
jupyter notebook student_loans_with_deep_learning.ipynb
```
