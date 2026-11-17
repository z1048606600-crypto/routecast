import csv
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from routecast.data import synthetic,adjacency,windows,read_csv,write_csv
from routecast.model import GraphTCN
from routecast.routing import plan
from routecast.api import app

def test_target_split_does_not_leak():
    x,_=synthetic(300);mean=x[:180].mean(0);std=x[:180].std(0)
    a,b,ts=windows(x,mean,std,180,240)
    assert ts.min()==180 and ts.max()+6<=240
    np.testing.assert_allclose(b[0],x[180:186])
    np.testing.assert_allclose(a[0],(x[148:180]-mean)/std)

def test_graph_batch_and_gradient():
    x,links=synthetic(100);m=GraphTCN(adjacency(links))
    y=m(torch.tensor(x[:64].reshape(2,32,9)))
    assert y.shape==(2,6,9)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_routing_allocates_capacity_across_demands():
    edges=[{'id':'a-b','source':'a','target':'b','capacity_mbps':100,'latency_ms':1}]
    result=plan(edges,[.5],[{'id':str(i),'source':'a','target':'b','mbps':25} for i in range(2)])
    assert [r['status'] for r in result['routes']]==['planned','blocked']
    assert result['remaining_capacity_mbps']['a-b']==15
    assert result['applied_to_devices'] is False

def test_route_selects_alternative_and_switch_penalty():
    edges=[{'id':i,'source':u,'target':v,'capacity_mbps':100,'latency_ms':w} for i,u,v,w in
           [('ab','a','b',1),('bc','b','c',1),('ac','a','c',3)]]
    d=[{'id':'d','source':'a','target':'c','mbps':5}]
    assert plan(edges,[.1,.1,.1],d)['routes'][0]['edges']==['ab','bc']
    assert plan(edges,[.1,.1,.1],d,previous={'d':['ac']})['routes'][0]['edges']==['ac']
    assert plan(edges,[.89,.89,.1],d)['routes'][0]['edges']==['ac']

def test_csv_roundtrip_and_missing_samples(tmp_path):
    x,links=synthetic(110);p=tmp_path/'x.csv';write_csv(p,x,links)
    np.testing.assert_allclose(read_csv(p,links),x,atol=1e-6)
    p.write_text(p.read_text().replace('0,r1-r2','0,missing',1))
    with pytest.raises(ValueError):read_csv(p,links)

def test_api_invalid_shape():
    with TestClient(app) as c:
        assert c.get('/health').status_code==200
        if c.get('/health').json()['model_ready']:
            assert c.post('/api/forecast',json={'history':[[.5]]*32}).status_code==422
            payload=c.get('/api/demo').json()
            assert c.post('/api/forecast',json={'history':payload['history']}).status_code==200
