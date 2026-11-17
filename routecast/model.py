import torch
from torch import nn

class GraphTCN(nn.Module):
    """Shared causal temporal encoder followed by normalized edge graph mixing."""
    def __init__(self, adjacency, horizon=6, hidden=24, graph=True):
        super().__init__()
        self.register_buffer('adjacency',torch.as_tensor(adjacency,dtype=torch.float32))
        self.graph = graph
        self.temporal = nn.Sequential(
            nn.Conv1d(1,hidden,5,dilation=1), nn.GELU(),
            nn.Conv1d(hidden,hidden,5,dilation=2), nn.GELU(),
            nn.Conv1d(hidden,hidden,3,dilation=4), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden*(2 if graph else 1),hidden),nn.GELU(),nn.Linear(hidden,horizon))

    def forward(self,x):
        b,t,e = x.shape
        h = self.temporal(x.permute(0,2,1).reshape(b*e,1,t))[:,:,-1].reshape(b,e,-1)
        if self.graph:
            neighbor = torch.einsum('ij,bjh->bih',self.adjacency,h)
            h = torch.cat([h,neighbor],dim=-1)
        # Residual over the latest load, in standardized coordinates.
        return self.head(h).permute(0,2,1)+x[:,-1:,:]
