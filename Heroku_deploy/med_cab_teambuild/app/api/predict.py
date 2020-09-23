from typing import Dict, Optional  # helps enforce typing
import logging
import random
import numpy as np
from fastapi import APIRouter
import joblib
import pandas as pd
from pydantic import BaseModel, Field, validator, Json


log = logging.getLogger(__name__)
router = APIRouter()

model = joblib.load("app/api/nn_model.joblib")

print("Serialized Model Loaded")

# cats = ['hybrid', 'sativa', 'indica', 'Aroused', 'Creative', 'Euphoric',
#         'Energetic', 'Euphoric', 'Focused', 'Giggly', 'Happy', 'Hungry',
#         'Relaxed', 'Sleepy', 'Talkative', 'Tingly', 'Uplifted', 'anxiety',
#         'depression', 'pain', 'fatigue', 'insomnia', 'brain fog',
#         'loss of appetite', 'nausea', 'low libido']

cats = ["body",
        "potent",
        "stress",
        "relaxing",
        "cerebral",
        "mind",
        "physical",
        "uplifting",
        "relaxation",
        "day",
        "cbd",
        "euphoria",
        "anxiety",
        "relief",
        "mood",
        "appetite",
        "mental",
        "depression",
        "energy",
        "balanced",
        "nausea",
        "creative",
        "insomnia",
        "alien",
        "good",
        "help",
        "stimulating",
        "pain",
        "fatigue",
        "brain fog",
        "loss of appetite",
        "low libido",
        "hybrid",
        "sativa",
        "indica",
        "Focused",
        "Happy",
        "Aroused",
        "Uplifted",
        "Creative",
        "Hungry",
        "Sleepy",
        "Giggly",
        "Relaxed",
        "Tingly",
        "Energetic",
        "Euphoric",
        "Talkative",
        "Grapefruit",
        "Pear",
        "Tree",
        "Tobacco",
        "Apple",
        "Herbal",
        "Citrus",
        "Sage",
        "Butter",
        "Bluberry",
        "Fruity",
        "Tree Fruit",
        "Rose",
        "Chestnut",
        "Skunk",
        "Pepper",
        "Fruit",
        "Apricot",
        "Mango",
        "Tea",
        "Vanilla",
        "Berry",
        "Strawberry",
        "Menthol",
        "Blue",
        "Honey",
        "Blueberry",
        "Minty",
        "Pine",
        "Lavender",
        "Flowery",
        "Orange",
        "Nutty",
        "Grapes",
        "Woody",
        "Tropical",
        "Peach",
        "Grape",
        "Diesel",
        "Spicy",
        "Mint",
        "Sweet",
        "Coffee",
        "Chemical",
        "Cheese",
        "Tar",
        "Ammonia",
        "Bubblegum",
        "Pineapple",
        "Lemon",
        "Plum",
        "Earthy",
        "Violet",
        "Pungent",
        "Lime"]

class UserInputData(BaseModel):
    """Create a class for OOP reasons.  I think we are better off hiding
    implementation instead of exposing the direct Dict structure
    also we will now get back more meaningful errors, 
    because fastapi is going to parse the new object for us,
    basically ensure a valid state for our object. """
    include: Optional[Dict[str, bool]]
    exclude: Optional[Dict[str, bool]]

    def to_df(self):
        '''somehow force shape, fillna'''
        df = pd.DataFrame(columns=cats)
        df.loc[0] = [0.5]*len(cats) # number of training dimensions; 0.5 is null
        for key, value in self.include.items(): # in 'include'
            df[key] = int(bool(value)) # converts T/F to ints 1/0
        return df


@router.post("/predict")
def predict_strain(user: UserInputData):
    """Predict the ideal strain based on user input"""
    X_new = user.to_df()
    log.info(X_new)
    neighbors = model.kneighbors(user.to_df())[1][0] # vid @ 56:02
    return {
        "IDs":[int(id_) for id_ in neighbors]
    }

# @router.get('/random')  # What is this route going to be?
# def random_penguin(): 
#     """Return a random penguin species"""
#     return random.choice(["Adelie", "ChinStrap", "Gentoo"])


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
