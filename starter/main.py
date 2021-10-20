# -*- coding: utf-8 -*-

"""
RESTful API using FastAPI with get and post methods.
"""

import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference, ModelConfig

mc = ModelConfig()
model = pickle.load(open(os.path.join(mc.model_path, 'gb_model.pkl'), 'rb'))
encoder = pickle.load(open(os.path.join(mc.model_path, 'one_hot_encoder.pkl'), 'rb'))
lb = pickle.load(open(os.path.join(mc.model_path, 'label_binarizer.pkl'), 'rb'))


class InputFeatures(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')


app = FastAPI()


@app.get("/")
async def welcome_msg():
    """Get welcome message."""
    return {'message': "Welcome to the API!"}


@app.post("/inference")
async def model_inference(input_features: InputFeatures):
    """Run model in inference mode."""
    input_df = pd.DataFrame([input_features.dict(by_alias=True)])
    X, _, _, _ = process_data(input_df, mc.cat_features, training=False,
                              encoder=encoder, lb=lb)
    prediction = inference(model, X)
    prediction = lb.inverse_transform(prediction).item(0)
    return {'prediction': prediction}
