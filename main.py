from fastapi import FastAPI, HTTPException
from api_datatypes import PersonData
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference, load_component
from starter.ml.constants import cat_features

model = None
encoder = None
try:
    model = load_component("model/model.pkl")
    encoder = load_component("model/encoder.pkl")
except:
    pass


title = "Income Prediction"
description = "This API helps to predict if the salary for the given information surpasses 50K"
version = "1.0.0"

app = FastAPI(title=title, description=description, version=version)


@app.get("/")
async def hello():
    return {"Hello, this API helps predict if the income is greater than 50K"}


@app.post("/predict/")
async def predict(personData: PersonData):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Sorry, there is not a model loaded on the server to process this request.")
    if encoder is None:
        raise HTTPException(status_code=500, detail=f"Sorry, currently there is not a encoder available one the server to process this request.")
    try:
        request_dict = personData.model_dump(by_alias=True)
        request_df = pd.DataFrame([request_dict])
        X, _, _, _ = process_data(
            request_df,
            categorical_features=cat_features,
            encoder=encoder, training=False
        )
        pred = inference(model, X)
        return {"prediction": bool(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Sorry, error during processing the request:\n{e}")

