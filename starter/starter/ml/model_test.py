# -*- coding: utf-8 -*-

"""
Set of tests for functions available in starter.ml.model module.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .data import process_data
from .model import compute_model_metrics, inference, ModelConfig, train_model

mc = ModelConfig()


@pytest.fixture
def get_train_data():
    train_data = pd.read_csv(os.path.join(mc.data_path, 'census_train.csv'))
    X_train, y_train, _, _ = process_data(
        train_data, categorical_features=mc.cat_features, label="salary", 
        training=True
    )
    return X_train, y_train


@pytest.fixture
def get_test_data():
    test_data = pd.read_csv(os.path.join(mc.data_path, 'census_test.csv'))
    with open(os.path.join(mc.model_path, 'one_hot_encoder.pkl'), 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    with open(os.path.join(mc.model_path, 'label_binarizer.pkl'), 'rb') as lb_file:
        lb = pickle.load(lb_file)
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=mc.cat_features, label="salary", 
        training=False, encoder=encoder, lb=lb
    )
    return X_test, y_test


@pytest.fixture
def get_model():
    with open(os.path.join(mc.model_path, 'gb_model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def test_train_model(get_train_data):
    """
    Test whether train_model function returns fitted instance of 
    GradientBoostingClassifier with two classes.
    """
    model = train_model(*get_train_data, hp_iter=1)
    
    assert isinstance(model, GradientBoostingClassifier)
    
    is_fitted = False
    try: 
        check_is_fitted(model)
        is_fitted = True
    except NotFittedError:
        pass
    assert is_fitted
    
    assert model.n_classes_ == 2


def test_inference(get_test_data, get_model):
    """
    Test whether inference function returns a numpy array of the same shape
    as y and values of 0 or 1.
    """
    X_test, y_test = get_test_data
    pred = inference(get_model, X_test)
    
    assert isinstance(pred, np.ndarray)
    assert pred.shape == y_test.shape
    assert set(pred) == set([0, 1])


def test_compute_model_metrics(get_test_data, get_model):
    """
    Test whether compute_model_metrics returns three metrics of float type
    and fbeta score exceeds 0.50 for saved model on test dataset.
    """
    X_test, y_test = get_test_data
    metrics = compute_model_metrics(y_test, inference(get_model, X_test))
    fbeta = metrics[2]

    assert len(metrics) == 3
    assert all(isinstance(metric, float) for metric in metrics)
    assert fbeta > 0.5
