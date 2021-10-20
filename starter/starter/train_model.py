# -*- coding: utf-8 -*-

"""
Script to train machine learning model.
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import get_slices_performance, inference, ModelConfig, train_model

mc = ModelConfig()

# Add code to load in the data.
data = pd.read_csv(os.path.join(mc.data_path, 'census_cleaned.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=0)
train.to_csv(os.path.join(mc.data_path, 'census_train.csv'), index=False)
test.to_csv(os.path.join(mc.data_path, 'census_test.csv'), index=False)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=mc.cat_features, label="salary", training=True
)

# Save model related artifacts.
with open(os.path.join(mc.model_path, 'one_hot_encoder.pkl'), 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)
with open(os.path.join(mc.model_path, 'label_binarizer.pkl'), 'wb') as lb_file:
    pickle.dump(lb, lb_file)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=mc.cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
classifier = train_model(X_train, y_train, hp_iter=50)
with open(os.path.join(mc.model_path, 'gb_model.pkl'), 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Evaluate overall model score and performance on slices.
perf_args = {
    'model': classifier,
    'data': train,
    'categorical_features': mc.cat_features,
    'process_data_fn': process_data,
    'one_hot_encoder': encoder,
    'label_binarizer': lb,
    'label': 'salary',
    'type': 'train'
}
perf_train = get_slices_performance(**perf_args)
perf_args['data'], perf_args['type'] = test, 'test'
perf_test = get_slices_performance(**perf_args)
perf = pd.concat([perf_train, perf_test])
perf.to_csv(os.path.join(mc.data_path, 'performance.csv'), index=False)
perf.to_csv(os.path.join('files_for_review', 'slice_output.txt'), index=False)

# Save samples for inference testing
pred = lb.inverse_transform(inference(classifier, X_test)).tolist()
test['pred'] = pred
pos_sample = test[test.pred == '>50K'].drop('salary', axis=1).sample(1)
neg_sample = test[test.pred == '<=50K'].drop('salary', axis=1).sample(1)
pd.concat([pos_sample, neg_sample]).to_csv(
    os.path.join(mc.data_path, 'samples_for_inference.csv'), index=False)
