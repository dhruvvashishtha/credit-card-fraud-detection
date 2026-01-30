# Credit Card Transaction Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset. The objective is to build, evaluate, and compare multiple classification models while applying probability threshold optimization to balance fraud detection coverage and false positive control.

The project follows an end-to-end modeling workflow, starting from baseline linear models to advanced tree-based and gradient boosting models, with a strong emphasis on evaluation metrics appropriate for imbalanced classification problems.

---

## Problem Statement
Credit card fraud detection is a critical real-world problem characterized by extreme class imbalance, where fraudulent transactions represent a very small fraction of total transactions. Traditional accuracy-based evaluation is misleading in such scenarios.

The goal of this project is to:
- Accurately identify fraudulent transactions
- Minimize false positives to avoid unnecessary alerts
- Select a model that generalizes well to unseen data

---

## Dataset
The dataset used in this project is sourced from Kaggle:

**Credit Card Fraud Detection Dataset**  
https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset contains transactional, temporal, and customer-related features along with a binary fraud label. Due to size and licensing considerations, the dataset is not included in this repository and must be downloaded separately from Kaggle.

---

## Approach

### Key Challenges
- Extreme class imbalance
- High cost of false positives
- Risk of overfitting with complex models

### Strategy
- Use appropriate evaluation metrics such as **average precision**, **F1-score**, **balanced accuracy**, and **ROC-AUC**
- Apply **probability threshold optimization** instead of using a fixed 0.5 cutoff
- Compare baseline and optimized versions of multiple models
- Evaluate performance separately on training, validation, and test datasets

---

## Models Evaluated

### Linear Models
- Logistic Regression (Baseline)
- Logistic Regression (Optimized with regularization and class weighting)

### Tree-Based Models
- Random Forest (Baseline)
- Random Forest (Optimized)

### Gradient Boosting Models
- XGBoost (Baseline)
- XGBoost (Optimized)
- LightGBM (Baseline)
- LightGBM (Optimized)

---

## Threshold Optimization
Instead of relying on the default probability threshold of 0.5, an optimized threshold is selected using the precision–recall curve on the training data. The threshold selection process prioritizes achieving high precision while maximizing recall under that constraint.

This approach better aligns model predictions with real-world fraud detection objectives, where false positives are costly.

---

## Model Comparison and Selection
Models are compared using validation metrics, with particular emphasis on:
- Average Precision
- F1-score
- Balanced Accuracy
- ROC-AUC

While linear and random forest models provide useful baselines, gradient boosting models significantly outperform them in terms of fraud detection capability.

**Optimized XGBoost** is selected as the final model due to:
- Best balance between precision and recall on validation data
- Strong generalization performance
- High ranking capability as indicated by ROC-AUC
- Stable behavior compared to more aggressive ensemble models

---

## Final Results
The optimized XGBoost model demonstrates strong performance on the held-out test dataset, maintaining high fraud recall while keeping false positives under control. The results validate the effectiveness of gradient boosting combined with threshold optimization for imbalanced fraud detection problems.

---

## Repository Structure
