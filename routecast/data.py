"""Chronological link telemetry; scaling is fitted on training data only."""
import csv
import json
from pathlib import Path
import numpy as np

LINKS = [
    ('r1','r2'), ('r1','r3'), ('r2','r3'), ('r2','r4'),
    ('r3','r4'), ('r3','r5'), ('r4','r5'), ('r4','r6'), ('r5','r6'),
]

def topology():
    return [{'id':f'{a}-{b}', 'source':a, 'target':b,
             'capacity_mbps':100.0, 'latency_ms':float(2+i%4)}
            for i,(a,b) in enumerate(LINKS)]

def synthetic(steps=1600, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float32)
    shared = 0.12*np.sin(t*2*np.pi/48)+0.08*np.sin(t*2*np.pi/170)
    x = np.stack([0.40+shared+0.12*np.sin(t*2*np.pi/48+i*.4)
                  +0.05*np.sin(t*2*np.pi/(23+i))+rng.normal(0,.018,steps)
                  for i in range(len(LINKS))], axis=1).astype('float32')
    # Short congestion episodes affect adjacent links at staggered times.
    for start in range(120,steps-40,150):
        for e in range(3):
            s = start+e*3
            x[s:s+24,e:e+3] += np.hanning(24)[:,None]*.28
    return np.clip(x,0,1.2), topology()

def write_csv(path, x, links):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','link_id','load_mbps','capacity_mbps'])
        for t,row in enumerate(x):
            for i,link in enumerate(links):
                writer.writerow([t,link['id'],round(float(row[i])*link['capacity_mbps'],5),link['capacity_mbps']])

def read_csv(path, links):
    ids = [e['id'] for e in links]
    rows = {}
    with open(path,encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            timestamp = float(row['timestamp'])
            if not np.isfinite(timestamp):
                raise ValueError('timestamps must be finite numeric seconds')
            lid = row['link_id']
            if lid not in ids:
                raise ValueError(f'unknown link {lid}')
            cap = float(row['capacity_mbps'])
            load = float(row['load_mbps'])
            expected = float(links[ids.index(lid)]['capacity_mbps'])
            if not np.isfinite([cap,load]).all() or cap<=0 or load<0 or not np.isclose(cap,expected):
                raise ValueError('invalid load or capacity differs from topology')
            bucket = rows.setdefault(timestamp,{})
            if lid in bucket:
                raise ValueError('duplicate timestamp/link')
            bucket[lid] = load/cap
    times = sorted(rows)
    if len(times)<100 or any(set(rows[t])!=set(ids) for t in times):
        raise ValueError('need >=100 complete timestamps')
    delta = np.diff(times)
    if not np.allclose(delta,delta[0]):
        raise ValueError('timestamps must be regularly sampled; resample before import')
    return np.array([[rows[t][lid] for lid in ids] for t in times],dtype='float32')

def adjacency(links):
    nodes = [{e['source'],e['target']} for e in links]
    a = np.array([[float(bool(x&y)) for y in nodes] for x in nodes],dtype='float32')
    return a/a.sum(axis=1,keepdims=True)

def windows(x, mean, std, start, stop, lookback=32, horizon=6):
    inputs,targets,ends = [],[],[]
    # Every target lies completely inside its split. Past context may precede it.
    for t in range(max(start,lookback),stop-horizon+1):
        inputs.append((x[t-lookback:t]-mean)/std)
        targets.append(x[t:t+horizon])
        ends.append(t)
    if not inputs:
        raise ValueError('split too short')
    return np.array(inputs),np.array(targets),np.array(ends)
