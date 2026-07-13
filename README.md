**Author:** Suhail Ahmed
**Date:** July 2026

# Credit Card Fraud Detection

## Problem
Detect fraudulent credit card transactions in a highly imbalanced dataset —
only 492 fraud cases out of 284,807 transactions (0.17%).

## Dataset
Download from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Approach
1. Explored class imbalance and transaction amount distribution
2. Scaled `Time` and `Amount` features (V1-V28 were pre-scaled by dataset creators)
3. Used stratified train-test split to preserve fraud ratio in both sets
4. Trained Logistic Regression — found accuracy (99.9%) was misleading due to imbalance
5. Tuned decision threshold (0.5 → 0.2) to improve fraud recall
6. Trained Random Forest for comparison
7. Tracked all experiments using MLflow

## Results

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| Logistic Regression (threshold=0.2) | 0.74 | 0.76 | 0.75 |
| **Random Forest** | **0.96** | **0.76** | **0.85** |

**Random Forest performed best**, catching the same proportion of fraud (76% recall)
with far fewer false alarms (96% precision vs 74%).

## Key Insight
Accuracy alone was misleading — a model predicting "not fraud" for everything would
still score 99.8% accuracy while catching zero fraud. Precision, recall, and F1-score
were the metrics that actually mattered for this imbalanced classification problem.

## Visualizations

**Class Imbalance**
![Class Distribution](Screenshot%202026-07-13%20123316.png)

**Transaction Amount Distribution**
![Amount Distribution](Screenshot%202026-07-13%20123337.png)

**Model Comparison (MLflow)**
![MLflow Comparison](Screenshot%202026-07-13%20133051.png)

## Tools Used
Python, Pandas, Seaborn, Scikit-learn, MLflow
