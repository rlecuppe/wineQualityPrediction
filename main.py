from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

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


@app.put("/api/model/")
async def addWine(wine: Wine):
    csvRow = f"{wine.fixed_acidity},{wine.volatile_acidity},{wine.citric_acid},{wine.residual_sugar},{wine.chlorides},{wine.free_sulfur_dioxide},{wine.total_sulfur_dioxide},{wine.density},{wine.pH},{wine.sulphates},{wine.alcohol},{wine.quality}"
    with open('./data/Wines.csv','a') as csvFile:
        csvFile.write(csvRow)
    return {"message": "Wine added"}

@app.get("/")
async def root():
    return {"message": "Hello World"}

