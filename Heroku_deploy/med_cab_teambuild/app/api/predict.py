from typing import Dict, Optional, List #, String# helps enforce typing
import random
import numpy as np
from fastapi import APIRouter
import joblib
import pandas as pd
from pydantic import BaseModel, Field, validator, Json
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# import en_core_web_sm
# from spacy import load
# nlp= en_core_web_sm.load()
# # tokenizer function
# def tokenizer(text):
#     doc=nlp(text)
#     return [token.lemma_ for token in doc if ((token.is_stop == False) and
#     (token.is_punct == False)) and (token.pos_ != 'PRON')]
# nlp_working = False
# nlp_preprocessing = TfidfVectorizer(stop_words = 'english',
#     ngram_range = (1, 2),
#     max_df = .95,
#     min_df = 3,
#     tokenizer = tokenizer)
# df = pd.read_csv('https://raw.githubusercontent.com/bw-med-cabinet-1/DS/master/data/cannabis_strain')
# df = df.drop('Unnamed: 0', axis= 1)
# nlp_preprocessing.fit_transform(df['effects'])
#print(f'preprocessing')
# dtm = pd.DataFrame(dtm.todense(), columns = nlp_preprocessing.get_feature_names())
dataframe = pd.read_csv('https://raw.githubusercontent.com/bw-med-cabinet-1/DS/master/data/Cannabis_Strains_Features.csv')
#print(len(dtm))
router = APIRouter()

nn_model = joblib.load("app/api/nn_model.joblib")
nlp_model = joblib.load("app/api/nlp_model.joblib")
#nlp_preprocessing = joblib.load("app/api/nlp_preprocessing.joblib")

print("Serialized Model Loaded")

nlp_cats = ['strain_id', 'strain', 'type', 'Rating', 'effects', 'flavor',
       'description']
# cats = ['hybrid', 'sativa', 'indica', 'Aroused', 'Creative', 'Euphoric',
#         'Energetic', 'Euphoric', 'Focused', 'Giggly', 'Happy', 'Hungry',
#         'Relaxed', 'Sleepy', 'Talkative', 'Tingly', 'Uplifted', 'anxiety',
#         'depression', 'pain', 'fatigue', 'insomnia', 'brain fog',
#         'loss of appetite', 'nausea', 'low libido']

nn_cats = ["body",
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
    include: Optional[List[str]]
    exclude: Optional[Dict[str, bool]]
    text: Optional[str]

    def categorical_formatting(self):
        '''somehow force shape, fillna'''
        df = pd.DataFrame(columns=nn_cats)
        df.loc[0] = [0.5]*len(nn_cats) # number of training dimensions; 0.5 is null
        for feature in self.include: # in 'include'
            df[feature] = 1 # converts T/F to ints 1/0
        return df
    
    def nlp_formatting(self):
        print(self.text)
        #vec = nlp_preprocessing.transform(self.text.encode('unicode_escape'))
        vec = nlp_preprocessing.transform([fR"{self.text}"])
        print(f'vec shape: {vec.shape}')
        # dense = vec.todense()
        # print(self.text)
        # print(f'dense: {dense}')
        # print(f'self.nlp_formatting() length:{len(dense)}')
        return vec


@router.post("/predict")
def predict_strain(user: UserInputData):
    """Predict the ideal strain based on user input"""
    nn_return_values = [] # initializing to empty for valid return
    nlp_return_value = []

    if user.include or user.exclude:
        X_new = user.categorical_formatting()
        print(X_new.shape)
        neighbors = nn_model.kneighbors(X_new)[1][0] # vid @ 56:02
        neighbor_ids = [int(id_) for id_ in neighbors]
        nn_return_values = [dataframe.iloc[id] for id in neighbor_ids]

    elif user.text and nlp_working:
        print(f'user.text = True')
        X_new = user.nlp_formatting()
        #vec = nlp_preprocessing.transform(X_new)
        dense = X_new.todense()
        print(f'dense/input shape : {dense.shape}')
        similar = nlp_model.kneighbors(dense, return_distance=False)
        similar.T
        output = []
        for i in range(5):
            elem = similar[0][i]
            output.append(elem)
        nlp_return_value = output[0]
        print(user.text)
        print(nlp_return_value)
    else: # if neither are given
        return {
            "error": "insufficient inputs"
        }
    return {
        "Nearest Neighbors": nn_return_values,
        "Text-based Prediction": nlp_return_value
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
