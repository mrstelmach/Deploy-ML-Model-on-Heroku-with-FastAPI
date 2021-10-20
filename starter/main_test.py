# -*- coding: utf-8 -*-

"""
Set of tests for API from main.py module.
"""

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from .main import app
from starter.starter.ml.model import ModelConfig

client = TestClient(app)
mc = ModelConfig()


@pytest.fixture
def get_samples():
    """Get random positive and negative samples."""
    samples = pd.read_csv(
        os.path.join(mc.data_path, 'samples_for_inference.csv'))
    pos_sample = samples[samples.pred == '>50K'].\
        drop('pred', axis=1).iloc[0].to_dict()
    neg_sample = samples[samples.pred == '<=50K'].\
        drop('pred', axis=1).iloc[0].to_dict()
    return pos_sample, neg_sample


def test_get_method():
    """Test get method with welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': "Welcome to the API!"}


def test_post_method_pos(get_samples):
    """Test post method for positive sample."""
    r = client.post(url="/inference", json=get_samples[0])
    assert r.status_code == 200
    assert r.json() == {'prediction': '>50K'}


def test_post_method_neg(get_samples):
    """Test post method for negative sample."""
    r = client.post(url="/inference", json=get_samples[1])
    assert r.status_code == 200
    assert r.json() == {'prediction': '<=50K'}
