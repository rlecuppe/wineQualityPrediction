from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import json
from fastapi.responses import FileResponse


app = FastAPI()

class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide : float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: Optional[int] = None


@app.get('/api/model/')
async def getModel():
    return FileResponse('./model/model.joblib')

@app.get('api/model/description')
async def decription():
    with open('./model/model.json', 'r') as f:
        model_dict = json.load(f)
    return model_dict

@app.put("/api/model/")
async def addWine(wine: Wine):
    csvRow = f"{wine.fixed_acidity},{wine.volatile_acidity},{wine.citric_acid},{wine.residual_sugar},{wine.chlorides},{wine.free_sulfur_dioxide},{wine.total_sulfur_dioxide},{wine.density},{wine.pH},{wine.sulphates},{wine.alcohol},{wine.quality}"
    with open('./data/Wines.csv','a') as csvFile:
        csvFile.write(csvRow)
    return {"message": "Wine added"}

@app.post("/api/predict/")
async def predictQuality(wine: Wine):
    rf = joblib.load('./model/model.joblib')
    df = pd.DataFrame([[wine.fixed_acidity,wine.volatile_acidity,wine.citric_acid,wine.residual_sugar,wine.chlorides,wine.free_sulfur_dioxide,wine.total_sulfur_dioxide,wine.density,wine.pH,wine.sulphates,wine.alcohol]],index=["1"], columns=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])
    pred = rf.predict(df)
    return {"quality": pred.item(0)}

@app.post("/api/model/retrain")
async def retrain():
    # Read the data
    wines = pd.read_csv("./data/Wines.csv")
    # Split the data
    wines = wines.drop("Id", axis = 1)
    wines_quality = wines["quality"]
    wines = wines.drop("quality", axis=1)
    x_train,x_test,y_train,y_test = train_test_split(wines, wines_quality, test_size=0.20, random_state=40)
    rf = RandomForestClassifier()
    rf = RandomForestClassifier(criterion = 'entropy', max_depth=8, max_features= 'log2', n_estimators=500)
    rf.fit(x_train, y_train)
    model_dict = rf.get_params()

    # Convert the dictionary to a JSON string
    model_json = json.dumps(model_dict)

    # Write the JSON string to a file
    with open('./model/model.json', 'w') as f:
        f.write(model_json)
    f.close()

    joblib.dump(rf, './model/model.joblib')
    return {"message": "Model retrained"}
