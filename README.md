# Network Intrusion Detection Project

## Overview
Welcome to my Network Intrusion Detection project! This project focuses on detecting intrusions in computer networks using machine learning techniques. I utilized Python, pandas, scikit-learn, and Flask for data preprocessing, model training, and creating a simple web application to showcase the results.

## Dataset
The dataset used for this project is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection). It contains various features related to network traffic and intrusion types, making it suitable for training machine learning models.

## Preprocessing
In this project, data preprocessing is a crucial step to prepare the dataset for training machine learning models. The `preprocessing.py` file contains a function named `preprocessing()` that defines the preprocessing steps for the dataset.

The preprocessing steps include:
- Encoding categorical columns ('protocol_type', 'service', and 'flag') using one-hot encoding.
- Scaling numerical columns ('src_bytes' and 'dst_bytes') using standardization.

These preprocessing steps ensure that the dataset is properly transformed and ready for training machine learning models.

## Training
The `train.py` file contains a function named `Train` that trains a machine learning model on the preprocessed data. It takes a DataFrame (`df`), a preprocessor, and a classifier as inputs. The function creates a pipeline that combines the preprocessor with the classifier, trains the pipeline on the training data, evaluates the model's performance using classification metrics, and saves the trained model to a file.

### Classifiers
Different classifiers were explored in this project:
- **Logistic Regression**: A simple linear classifier used for binary classification tasks.
- **Decision Trees**: Non-parametric models that partition the feature space into regions.
- **Random Forest**: Ensemble learning method that constructs multiple decision trees and aggregates their predictions.
- **Support Vector Machine (SVM)**: Powerful classifiers that find the hyperplane that best separates classes in the feature space.
- **Gradient Boosting**: Ensemble learning technique that builds a sequence of weak learners and combines them to create a strong learner.
- **K-Nearest Neighbors (KNN)**: Intuitive classifier that classifies new cases based on similarity measures with existing cases.

Each classifier has its strengths and weaknesses, and the choice often depends on the dataset characteristics and problem requirements.


