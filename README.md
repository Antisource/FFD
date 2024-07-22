# Financial Fraud Detection using Supervised Machine Learning Models

This repository contains code for detecting fraudulent credit card transactions. We have used PCA(Principal for dimensionality reduction and GAN (Generative Adversial Network) to balance the imbalanced dataset.
The following supervised learning models are used for fraud detection:
- Random Forest
- Support Vector Machine (SVM)
- K Nearest Neighbours (KNN)
- Naive Bayes
- XGBoost
- Light GBM

The above models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
  
## Generative Adversarial Network (GAN) Architecture

![GAN Architecture](https://developers.google.com/static/machine-learning/gan/images/gan_diagram.svg)

## Dataset

The dataset used in this project is a publicly available credit card transaction dataset containing a mix of genuine and fraudulent transactions. You can access the dataset on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Overview

The main steps involved in this project are as follows:

1. **Importing Dependencies**: Install required libraries and import necessary modules for data preprocessing, model building, and evaluation.

2. **Importing the Dataset**: Load the credit card transaction dataset and explore its structure.

3. **Data Preprocessing and Exploration**: Handle missing values, scale features, and explore the data distribution. Visualize the data using PCA.

4. **Building the Generator Model**: Defined the architecture of the generator model, which generates synthetic fraudulent data.

5. **Building the Discriminator Model**: Defined the architecture of the discriminator model, which classifies whether data is real or synthetic.

6. **Combining Generator and Discriminator to Build GAN**: Combined the generator and discriminator models to form a Generative Adversarial Network.

7. **Training and Evaluating the GAN**: Trained the GAN and monitor its performance using PCA. Generate synthetic data using the trained generator.

8. **Building the Fraud Detection Model**: Train various fraud detection models (Random Forest, SVM, KNN, Naive Bayes) using both real and synthetic data.

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/Antisource/FFD.git
   ```
2. Install dependencies:
  ```
   pip install tensorflow==2.11 visualkeras numpy pandas scikit-learn seaborn matplotlib plotly
   ```
   
   
