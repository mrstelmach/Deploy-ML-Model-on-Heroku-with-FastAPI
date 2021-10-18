# -*- coding: utf-8 -*-

"""
Set of tests for functions available in starter.ml.model module.
"""

import os

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .data import process_data
from .model import compute_model_metrics, inference, train_model

DATA_PATH = 'starter/data'

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


@pytest.fixture
def get_train_data():
    train_data = pd.read_csv(os.path.join(DATA_PATH, 'census_train.csv'))
    X_train, y_train, _, _ = process_data(
        train_data, categorical_features=cat_features, label="salary", 
        training=True
    )
    return X_train, y_train


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
