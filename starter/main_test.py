# -*- coding: utf-8 -*-

"""
Set of tests for API from main.py module.
"""

from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_get_method():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': "Welcome to the API!"}


test_get_method()