# -*- coding: utf-8 -*-

"""
RESTful API using FastAPI with get and post methods.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def welcome_msg():
    return {'message': "Welcome to the API!"}
