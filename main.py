from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
import numpy as np
import pandas as pd
from typing import Literal
import joblib

app=FastAPI(title='Credit Card Fraud Detection API')


class CardData(BaseModel):
    Time: float=Field(ge=0)
    V1:float
    V2:float
    V3:float
    V4:float
    V5:float
    V6:float
    V7:float
    V8:float
    V9:float
    V10:float
    V11:float
    V12:float
    V13:float
    V14:float
    V15:float
    V16:float
    V17:float
    V18:float
    V19:float
    V20:float
    V21:float
    V22:float
    V23:float
    V24:float
    V25:float
    V26:float
    V27:float
    V28:float
    Amount:float

    model_config=ConfigDict(populate_by_name=True)

class ClassificationResponse(BaseModel):
    is_anomaly: bool
    Fraud_Probability:float

class UnsupervisedResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    risk:str    


classification_scaler=joblib.load('Models/classification_scaler.pkl')
model_rf=joblib.load('Models/rf.pkl')
unsupervised_scaler=joblib.load('Models/unsupervised_scaler.pkl')
model_iso=joblib.load('Models/iso.pkl')

@app.get('/root')
def root():
    return {'message':'Welcome to Credit Card Fraud API'}

@app.get('/health')
def health_check():
    return {'status':'active'}


@app.post('/predict/classification')
def predict_classification(data:CardData):
    input_df=pd.DataFrame([data.model_dump(by_alias=True)])

    num_cols=['Time','V1','V2', 'V3', 'V4','V5','V6','V7','V8','V9','V10','V11','V12', 'V13','V14','V15','V16','V17','V18','V19',
 'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
    
    input_df[num_cols]=classification_scaler.transform(input_df[num_cols])
    prediction=model_rf.predict(input_df)[0]
    prob=model_rf.predict_proba(input_df)[0][1] 

    return ClassificationResponse(is_anomaly=prediction,Fraud_Probability=prob)


@app.post('/predict/unsupervised')
def predict_unsupervised(data:CardData):
    input_df=pd.DataFrame([data.model_dump(by_alias=True)])
    
    num_cols=['Time','V1','V2', 'V3', 'V4','V5','V6','V7','V8','V9','V10','V11','V12', 'V13','V14','V15','V16','V17','V18','V19',
 'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
    
    input_df[num_cols]=unsupervised_scaler.transform(input_df[num_cols])

    prediction=model_iso.predict(input_df)[0]
    score=model_iso.decision_function(input_df)[0]

    is_anomaly=(prediction==-1)

    if score<-0.2:
        risk_level='high'
    elif score<-0.1:
        risk_level='medium'
    elif score<=0:
        risk_level='low'
    else:
        risk_level='No Risk'    


    return UnsupervisedResponse(is_anomaly=is_anomaly,anomaly_score=score,risk=risk_level)