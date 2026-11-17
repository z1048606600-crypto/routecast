import csv
import json
import pytest
from routecast.ingest import convert

def test_snmp_counter_units_and_reset(tmp_path):
    p=tmp_path/'prom.json';m=tmp_path/'map.json';out=tmp_path/'telemetry.csv'
    data={'status':'success','data':{'resultType':'matrix','result':[{'metric':{'instance':'router1','ifIndex':'2'},'values':[[0,'1000'],[10,'12501000']]}]}}
    p.write_text(json.dumps(data));m.write_text(json.dumps({'router1/2':{'link_id':'a-b','capacity_mbps':100}}))
    assert convert(p,m,out)['timestamps']==1
    row=list(csv.DictReader(out.open()))[0]
    assert float(row['load_mbps'])==10
    data['data']['result'][0]['values'][1][1]='0';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='reset'):convert(p,m,out)
