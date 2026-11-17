from pathlib import Path
import numpy as np
import torch
from .model import GraphTCN
from .data import adjacency

class Forecaster:
    def __init__(self,path='artifacts/model.pt'):
        # Only load locally produced weights. Never accept checkpoints from HTTP.
        self.ckpt=torch.load(path,map_location='cpu',weights_only=True)
        self.links=self.ckpt['links']
        self.model=GraphTCN(adjacency(self.links),horizon=self.ckpt['horizon'])
        self.model.load_state_dict(self.ckpt['state']);self.model.eval()
        self.mean=self.ckpt['mean'];self.std=self.ckpt['std']

    def predict(self,history):
        x=np.asarray(history,dtype='float32')
        if x.shape!=(self.ckpt['lookback'],len(self.links)) or not np.isfinite(x).all() or (x<0).any() or (x>2).any():
            raise ValueError(f'expected {self.ckpt["lookback"]} x {len(self.links)} finite utilization values in [0,2]')
        with torch.inference_mode():
            y=self.model(((torch.tensor(x)-self.mean)/self.std)[None])[0]*self.std+self.mean
        return y.clamp(0,2).numpy()
