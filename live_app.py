# -*- coding: utf-8 -*-

"""
Live test of Heroku app.
"""

import json
import requests

sample_data = {
    'age': 40,
    'workclass': 'Private',
    'fnlgt': 102606,
    'education': 'Masters',
    'education-num': 14,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

r = requests.post(url="https://ml-model-with-fastapi.herokuapp.com/inference",
                  data=json.dumps(sample_data))
print(f'Status code: {r.status_code}')
print(f'App returns: {r.json()}')