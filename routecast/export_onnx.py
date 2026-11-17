import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .service import Forecaster

def export(output='artifacts'):
    import onnxruntime as ort
    f=Forecaster(Path(output)/'model.pt')
    x=torch.tensor((np.load(Path(output)/'demo_history.npy')-f.mean.numpy())/f.std.numpy())[None]
    path=Path(output)/'model.onnx'
    torch.onnx.export(f.model,x,str(path),input_names=['history_standardized'],output_names=['forecast_standardized'],
                      dynamic_axes={'history_standardized':{0:'batch'},'forecast_standardized':{0:'batch'}},opset_version=17)
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(9);errors=[]
    for batch in [1,4]:
        values=rng.normal(size=(batch,32,len(f.links))).astype('float32')
        with torch.inference_mode():native=f.model(torch.tensor(values)).numpy()
        exported=session.run(None,{'history_standardized':values})[0]
        errors.append(float(np.abs(native-exported).max()))
    report={'opset':17,'max_absolute_error':max(errors),'tested_batch_sizes':[1,4],
            'input':'standardized utilization; apply mean/std from preprocessing.json before/after inference'}
    if report['max_absolute_error']>1e-4:raise ValueError('ONNX parity check failed')
    Path(output,'onnx_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    Path(output,'preprocessing.json').write_text(json.dumps({'mean':f.mean.tolist(),'std':f.std.tolist(),'links':f.links}),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts');a=p.parse_args();export(a.output)
