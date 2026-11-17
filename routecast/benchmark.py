"""Replay independent demand intervals against held-out actual background load."""
import json
from pathlib import Path
import numpy as np
from .data import synthetic,windows
from .service import Forecaster
from .routing import plan

def benchmark(output='artifacts'):
    f=Forecaster(Path(output)/'model.pt');x,links=synthetic();start=int(len(x)*.8)
    demands=[{'id':'video','source':'r1','target':'r6','mbps':15},
             {'id':'voice','source':'r2','target':'r5','mbps':5}]
    reports={}
    for mode in ['shortest_latency','persistence','graph_tcn']:
        admitted=violations=switches=opportunities=0;previous={};costs=[]
        for t in range(start,len(x)-6,6):
            history=x[t-32:t]
            predicted=(f.predict(history).max(0) if mode=='graph_tcn' else history[-1] if mode=='persistence' else np.zeros(len(links)))
            planned=plan(links,predicted.tolist(),demands,previous=previous)
            actual=x[t:t+6].max(0);allocation={e['id']:0. for e in links}
            for route in planned['routes']:
                opportunities+=1
                if route['status']!='planned':continue
                admitted+=1;costs.append(route['cost'])
                if route['id'] in previous and route['edges']!=previous[route['id']]:switches+=1
                previous[route['id']]=route['edges']
                for edge in route['edges']:allocation[edge]+=route['mbps']
            violated={e['id'] for i,e in enumerate(links) if actual[i]+allocation[e['id']]/e['capacity_mbps']>.9+1e-6}
            for route in planned['routes']:
                if route['status']=='planned' and set(route['edges'])&violated:violations+=1
        reports[mode]={'admitted':admitted,'opportunities':opportunities,'admission_rate':admitted/opportunities,
                       'admitted_intervals_exceeding_90pct':violations,'violation_rate_of_admitted':violations/max(admitted,1),
                       'route_switches':switches}
    report={'data_source':'synthetic_seed_42_test_interval_only','scope':'simulation_no_packet_level_network_emulator',
            'note':'Each interval has a fresh demand allocation. Background loads are exogenous, shared undirected link capacities. Not a proof of real network gains.',
            'results':reports}
    Path(output,'routing_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':benchmark()
