# Insurance Cross-Selling Prediction

This project is part of a Kaggle competition to predict cross-selling opportunities in the insurance industry. The goal is to build a model that can predict whether a customer is likely to purchase an additional insurance product based on their profile.

## Project Overview

The project involves the following steps:
1. **Data Loading:** Load the training and test datasets.
2. **Data Preprocessing:** Handle missing values, encode categorical variables, and perform feature engineering.
3. **Model Building:** Train machine learning models such as CatBoost, XGBoost, and Ridge Classifier.
4. **Model Evaluation:** Evaluate the model using cross-validation techniques like Stratified K-Fold.
5. **Prediction:** Generate predictions for the test dataset.
6. **Submission:** Prepare and submit the results to Kaggle.

## Dataset

The dataset for this competition can be found on the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s4e7/data). It includes a training set, a test set, and a sample submission file.

## Dataset Description
Variable Definition:

id -> Unique ID for the customer
Gender ->Gender of the customer
Age ->Age of the customer
Driving_License ->0 : Customer does not have DL, 1 : Customer already has DL
Region_Code ->Unique code for the region of the customer
Previously_Insured ->1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance
Vehicle_Age ->Age of the Vehicle
Vehicle_Damage ->1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.
Annual_Premium ->The amount customer needs to pay as premium in the year
Policy_Sales_Channel ->Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
Vintage ->Number of Days, Customer has been associated with the company
Response ->1 : Customer is interested, 0 : Customer is not interested


### Prerequisites

Here is a list of all the libraries used in this notebook:

1. **numpy** (`import numpy as np`)
2. **pandas** (`import pandas as pd`)
3. **catboost** (`from catboost import CatBoostClassifier, Pool`)
4. **xgboost** (`from xgboost import XGBClassifier`)
5. **sklearn** (from `sklearn.model_selection`, `sklearn.metrics`, `sklearn.utils`)
6. **category_encoders** (`from category_encoders import TargetEncoder, MEstimateEncoder`)
7. **warnings** (`import warnings`)
8. **gc** (`import gc`)
9. **torch** (`import torch`, `from torch import nn`, `from torch.utils.data import DataLoader, TensorDataset`)
10. **torch.optim** (`import torch.optim as optim`)
11. **torch.optim.lr_scheduler** (`from torch.optim.lr_scheduler import ReduceLROnPlateau`)
12. **polars** (`import polars as pl`)
13. **optuna** (`import optuna`)

Note: The `openfe` library is commented out, so it’s not currently being used in the code.
