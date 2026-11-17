"""Pure planning: no router credentials, device changes or network I/O."""
import heapq
import math

def plan(links, forecast, demands, headroom=.90, previous=None, switch_penalty_ms=4.):
    if not 0<headroom<=1:
        raise ValueError('headroom must be in (0,1]')
    if len(forecast)!=len(links) or any(not math.isfinite(x) or x<0 for x in forecast):
        raise ValueError('forecast must contain one nonnegative finite load per link')
    if len({e['id'] for e in links})!=len(links):
        raise ValueError('duplicate link ID')
    remaining = {e['id']:max(0,e['capacity_mbps']*(headroom-forecast[i])) for i,e in enumerate(links)}
    graph = {}
    nodes=set()
    for i,e in enumerate(links):
        if e['capacity_mbps']<=0 or e['latency_ms']<0:
            raise ValueError('invalid topology')
        nodes.update([e['source'],e['target']])
        for u,v in [(e['source'],e['target']),(e['target'],e['source'])]:
            graph.setdefault(u,[]).append((v,e,i))
    output=[]
    previous=previous or {}
    if len({d['id'] for d in demands})!=len(demands):
        raise ValueError('duplicate demand ID')
    # Greedy unsplittable multicommodity approximation, largest demand first.
    for d in sorted(demands,key=lambda d:-d['mbps']):
        if d['source'] not in nodes or d['target'] not in nodes or not math.isfinite(d['mbps']) or d['mbps']<=0:
            raise ValueError('invalid demand')
        q=[(0.,d['source'],[],[d['source']])]
        best={}
        chosen=None
        while q:
            cost,u,edges,path=heapq.heappop(q)
            if u in best:
                continue
            best[u]=cost
            if u==d['target']:
                chosen=(cost,edges,path)
                break
            for v,e,i in graph.get(u,[]):
                if v in best or remaining[e['id']]+1e-6<d['mbps']:
                    continue
                penalty = switch_penalty_ms if d['id'] in previous and e['id'] not in previous[d['id']] else 0
                weight = e['latency_ms']*(1+3*forecast[i]**2)+penalty
                heapq.heappush(q,(cost+weight,v,edges+[e['id']],path+[v]))
        if chosen is None:
            output.append({'id':d['id'],'status':'blocked','reason':'insufficient residual capacity'})
        else:
            cost,edges,path=chosen
            for eid in edges:
                remaining[eid]-=d['mbps']
            output.append({'id':d['id'],'status':'planned','path':path,'edges':edges,'mbps':d['mbps'],'cost':round(cost,3)})
    return {'routes':output,'remaining_capacity_mbps':remaining,'strategy':'greedy_capacity_constrained','applied_to_devices':False}
