"""Convert exported Prometheus SNMP counter samples to the training CSV contract."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def convert(input_json,mapping_json,output_csv):
    payload=json.loads(Path(input_json).read_text(encoding='utf-8'))
    mapping=json.loads(Path(mapping_json).read_text(encoding='utf-8'))
    if payload.get('status')!='success' or payload['data']['resultType']!='matrix':
        raise ValueError('expected Prometheus query_range matrix response')
    rows={};seen=set()
    for result in payload['data']['result']:
        identity=result['metric'].get('instance','')+'/'+result['metric'].get('ifIndex','')
        if identity not in mapping:continue
        link=mapping[identity];lid=link['link_id'];cap=float(link['capacity_mbps'])
        if lid in seen:raise ValueError('multiple series map to the same link')
        seen.add(lid)
        values=sorted((float(t),float(v)) for t,v in result['values'])
        for (previous_t,previous_v),(t,v) in zip(values,values[1:]):
            if not np.isfinite([previous_t,previous_v,t,v,cap]).all() or cap<=0 or t<=previous_t:
                raise ValueError('invalid counter sample')
            if v<previous_v:raise ValueError(f'counter reset for {lid}; split/resample explicitly before training')
            mbps=(v-previous_v)*8/(t-previous_t)/1e6
            bucket=rows.setdefault(t,{})
            if lid in bucket:raise ValueError('duplicate link timestamp')
            bucket[lid]=(mbps,cap)
    required={entry['link_id'] for entry in mapping.values()}
    if not required or seen!=required or any(set(r)!=required for r in rows.values()):
        raise ValueError('incomplete link coverage; no implicit missing-value fill')
    with open(output_csv,'w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['timestamp','link_id','load_mbps','capacity_mbps'])
        for t in sorted(rows):
            for lid,(load,cap) in sorted(rows[t].items()):writer.writerow([t,lid,load,cap])
    return {'timestamps':len(rows),'links':len(required),'units':'Mbps','counter_metric':'ifHCOutOctets'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prometheus_json');p.add_argument('mapping_json');p.add_argument('output_csv')
    a=p.parse_args();print(json.dumps(convert(a.prometheus_json,a.mapping_json,a.output_csv),indent=2))
