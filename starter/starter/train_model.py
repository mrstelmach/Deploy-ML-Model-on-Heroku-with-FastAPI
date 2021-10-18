# -*- coding: utf-8 -*-

"""
Script to train machine learning model.
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, inference, train_model 

DATA_PATH = 'starter/data'
MODEL_PATH = 'starter/model'

# Add code to load in the data.
data = pd.read_csv(os.path.join(DATA_PATH, 'census_cleaned.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
train.to_csv(os.path.join(DATA_PATH, 'census_train.csv'), index=False)
test.to_csv(os.path.join(DATA_PATH, 'census_test.csv'), index=False)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save model related artifacts
with open(os.path.join(MODEL_PATH, 'one_hot_encoder.pkl'), 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)
with open(os.path.join(MODEL_PATH, 'label_binarizer.pkl'), 'wb') as lb_file:
    pickle.dump(lb, lb_file)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
classifier = train_model(X_train, y_train, hp_iter=25)
with open(os.path.join(MODEL_PATH, 'gb_model.pkl'), 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Evaluate model
print(compute_model_metrics(y_train, inference(classifier, X_train)))
print(compute_model_metrics(y_test, inference(classifier, X_test)))
