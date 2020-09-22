from typing import Dict  # helps enforce typing
import logging
import random

from fastapi import APIRouter
import joblib
import pandas as pd
from pydantic import BaseModel, Field, validator


log = logging.getLogger(__name__)
router = APIRouter()

model = joblib.load("app/api/model.joblib")

print("Serialized Model Loaded")


# create a class for OOP reasons.  I think we are better off hiding
# implementation instead of exposing the direct Dict structure
# also we will now get back more meaningful errors, 
# because fastapi is going to parse the new object for us,
# basically ensure a valid state for our object. 

class UserInputData(BaseModel):
    """Create a class for OOP reasons.  I think we are better off hiding
    implementation instead of exposing the direct Dict structure
    also we will now get back more meaningful errors, 
    because fastapi is going to parse the new object for us,
    basically ensure a valid state for our object. """

    ailment: str
    primary_impact: str
    undesired_impact: str

    def to_df(self):
        return pd.DataFrame([dict(self)])


@router.post("/predict")
def predict_strain(user: UserInputData):
    """Predict the ideal strain based on user input"""
    df = user.to_df()
    # this is where the model goes
    # ideal_strain[0] = model.predict(df)
    return df.shape


@router.get('/random')  # What is this route going to be?
def random_penguin(): 
    """Return a random penguin species"""
    return random.choice(["Adelie", "ChinStrap", "Gentoo"])


# class Item(BaseModel):
#     """Use this data model to parse the request body JSON."""

#     x1: float = Field(..., example=3.14)
#     x2: int = Field(..., example=-42)
#     x3: str = Field(..., example='banjo')

#     def to_df(self):
#         """Convert pydantic object to pandas dataframe with 1 row."""
#         return pd.DataFrame([dict(self)])

#     @validator('x1')
#     def x1_must_be_positive(cls, value):
#         """Validate that x1 is a positive number."""
#         assert value > 0, f'x1 == {value}, must be > 0'
#         return value


# @router.post('/predict')
# async def predict(item: Item):
#     """
#     Make random baseline predictions for classification problem 🔮

#     ### Request Body
#     - `x1`: positive float
#     - `x2`: integer
#     - `x3`: string

#     ### Response
#     - `prediction`: boolean, at random
#     - `predict_proba`: float between 0.5 and 1.0, 
#     representing the predicted class's probability

#     Replace the placeholder docstring and fake predictions with your own model.
#     """

#     X_new = item.to_df()
#     log.info(X_new)
#     y_pred = random.choice([True, False])
#     y_pred_proba = random.random() / 2 + 0.5
#     return {
#         'prediction': y_pred,
#         'probability': y_pred_proba
#     }
