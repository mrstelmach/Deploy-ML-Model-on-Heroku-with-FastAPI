# -*- coding: utf-8 -*-

"""
RESTful API using FastAPI with get and post methods.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field


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
    return {'message': "Welcome to the API!"}


@app.post("/inference")
async def model_inference(input_features: InputFeatures):
    pass