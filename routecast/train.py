import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .data import synthetic,topology,adjacency,windows,write_csv,read_csv
from .model import GraphTCN

def metrics(pred,target):
    return {'mae_utilization':float(np.abs(pred-target).mean()),
            'rmse_utilization':float(np.sqrt(((pred-target)**2).mean()))}

def train(output='artifacts',epochs=18,csv_path=None,topology_path=None):
    torch.set_num_threads(2)
    np.random.seed(42)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if csv_path:
        if not topology_path:
            raise ValueError('--topology is required with --csv')
        links=json.loads(Path(topology_path).read_text(encoding='utf-8'))
        x=read_csv(csv_path,links)
        source='user_imported_csv'
    else:
        x,links=synthetic();source='synthetic_network_telemetry_seed_42'
    n=len(x); tr=int(n*.65); va=int(n*.8)
    mean=x[:tr].mean(axis=0);std=x[:tr].std(axis=0).clip(.01)
    splits=[windows(x,mean,std,a,b) for a,b in [(32,tr),(tr,va),(va,n)]]
    datasets=[(torch.tensor(a),torch.tensor((b-mean)/std)) for a,b,_ in splits]
    report={'data_source':source,'samples':n,'seed':42,'split':{'train':[0,tr],'validation':[tr,va],'test':[va,n]},
            'lookback':32,'horizon':6,'epochs':epochs,'torch':torch.__version__,'platform':platform.platform(),
            'data_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'models':{}}
    for graph,name in [(False,'temporal_only'),(True,'graph_tcn')]:
        torch.manual_seed(42)
        model=GraphTCN(adjacency(links),graph=graph)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
        best=float('inf');state=None;history=[];started=time.perf_counter()
        for epoch in range(epochs):
            model.train();a,b=datasets[0]
            for ids in torch.randperm(len(a)).split(64):
                optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(a[ids]),b[ids]);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val=float(torch.nn.functional.mse_loss(model(datasets[1][0]),datasets[1][1]))
            history.append(val)
            if val<best:
                best=val;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            started_infer=time.perf_counter();pred=model(datasets[2][0]).numpy()*std+mean
            batch_ms=(time.perf_counter()-started_infer)*1000
        report['models'][name]={**metrics(pred,splits[2][1]),'train_seconds':time.perf_counter()-started,
                               'test_batch_ms':batch_ms,'validation_mse':history,'parameters':sum(p.numel() for p in model.parameters())}
        if graph:
            torch.save({'state':state,'mean':torch.tensor(mean),'std':torch.tensor(std),'links':links,
                        'horizon':6,'lookback':32,'source':source},out/'model.pt')
            np.save(out/'demo_history.npy',x[-32:])
            np.save(out/'test_predictions.npy',pred)
    test_y=splits[2][1];times=splits[2][2]
    persistence=np.stack([np.tile(x[t-1],(6,1)) for t in times])
    seasonal=np.stack([x[t-48:t-48+6] for t in times])
    report['models']['persistence']=metrics(persistence,test_y)
    report['models']['seasonal_48']=metrics(seasonal,test_y)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'topology.json').write_text(json.dumps(links,indent=2),encoding='utf-8')
    write_csv(out/'demo_telemetry.csv',x[-120:],links)
    print(json.dumps({k:{m:round(v,5) for m,v in val.items() if isinstance(v,(int,float))} for k,val in report['models'].items()},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts');p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--csv');p.add_argument('--topology');args=p.parse_args()
    train(args.output,args.epochs,args.csv,args.topology)
