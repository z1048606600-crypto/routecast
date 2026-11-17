import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field,ConfigDict
from .service import Forecaster
from .routing import plan

ARTIFACTS=Path(os.getenv('ARTIFACT_DIR','artifacts'))
@asynccontextmanager
async def lifespan(app):
    app.state.forecaster=Forecaster(ARTIFACTS/'model.pt') if (ARTIFACTS/'model.pt').exists() else None
    yield

app=FastAPI(title='RouteCast',version='0.1.0',lifespan=lifespan)

class Demand(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    id:str=Field(min_length=1,max_length=80)
    source:str
    target:str
    mbps:float=Field(gt=0,le=10000)

class ForecastRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    history:list[list[float]]=Field(min_length=32,max_length=32)
    demands:list[Demand]=Field(default_factory=list,max_length=100)
    previous:dict[str,list[str]]=Field(default_factory=dict)

@app.get('/',response_class=HTMLResponse)
def home():
    return Path(__file__).with_name('web').joinpath('index.html').read_text(encoding='utf-8')

@app.get('/health')
def health():
    return {'status':'ok','model_ready':app.state.forecaster is not None}

@app.get('/api/demo')
def demo():
    if app.state.forecaster is None:
        raise HTTPException(503,'run python -m routecast.train first')
    return {'history':np.load(ARTIFACTS/'demo_history.npy').tolist(),
            'links':app.state.forecaster.links,'metrics':json.loads((ARTIFACTS/'metrics.json').read_text())}

@app.post('/api/forecast')
def forecast(req:ForecastRequest):
    f=app.state.forecaster
    if f is None:
        raise HTTPException(503,'model is not trained')
    try:
        y=f.predict(req.history)
        result=plan(f.links,y.max(axis=0).tolist(),[d.model_dump() for d in req.demands],previous=req.previous)
        return {'forecast':y.tolist(),'links':f.links,'planning':result,'model_source':f.ckpt['source']}
    except ValueError as e:
        raise HTTPException(422,str(e)) from e
