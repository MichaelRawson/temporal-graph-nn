import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear
import torch.nn.functional as F

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

LAYERS = 4

class T1Layer(Module):
    """a single layer in a T1 model"""

    save_u: Tensor
    """(stale, non-differentiable) h_v(t') at previous layer if {u, v} is an edge in the temporal graph at t'"""
    save_u: Tensor
    """(stale, non-differentiable) h_u(t') at previous layer if {u, v} is an edge in the temporal graph at t'"""
    agg: Tensor
    """temporary buffer for aggregation results"""
    src: Tensor
    """temporary buffer for scatter source"""
    w1: Linear
    """first linear transform"""
    w2: Linear
    """second linear transform"""

    def __init__(self, total_nodes: int, total_events: int, previous_embed_size: int):
        super().__init__()
        self.register_buffer('save_u', torch.zeros(total_events, previous_embed_size), persistent=False)
        self.register_buffer('save_v', torch.zeros(total_events, previous_embed_size), persistent=False)
        agg_size = previous_embed_size + 1
        self.register_buffer('agg', torch.zeros(total_nodes, agg_size), persistent=False)
        self.register_buffer('src', torch.zeros(total_events, previous_embed_size + 1), persistent=False)
        self.w1 = Linear(agg_size, agg_size)
        out_size = agg_size + previous_embed_size
        self.w2 = Linear(out_size, out_size)

    def save(self, h_u: Tensor, h_v: Tensor, event: int):
        """remember h_v(t) and h_u(t) for future reference"""
        self.save_u[event] = h_u
        self.save_v[event] = h_v

    def forward(self, u: Tensor, v: Tensor, g: Tensor, h: Tensor, tick: int) -> Tensor:
        """forwards pass"""

        # move dimensions around a bit, should be cheap
        u = u.unsqueeze(1).expand(-1, self.agg.shape[1])
        v = v.unsqueeze(1).expand(-1, self.agg.shape[1])
        save_u = self.save_u[:tick]
        save_v = self.save_v[:tick]

        # zero aggregation
        self.agg.zero_()
        # only care about the first `tick` entries of `self.src`
        src = self.src[:tick]

        # aggregate into v
        src[:,:-1] = save_u
        src[:, -1] = g
        self.agg.scatter_add_(0, v, src)
        # aggregate into u
        src[:tick,:-1] = save_v
        src[:tick, -1] = g
        self.agg.scatter_add_(0, u, src)

        return self.w2(torch.cat((h, F.relu(self.w1(self.agg))), 1))

class T1(Module):
    """a T1 model"""

    total_nodes: int
    """total nodes in the graph"""
    tick: int
    """current event number"""
    layers: ModuleList
    """the layers for this model"""

    def __init__(self, total_nodes: int, total_events: int):
        super().__init__()
        self.total_nodes = total_nodes
        self.tick = 0

        layers = []
        embed_size = 0
        for _ in range(LAYERS):
            layers.append(T1Layer(total_nodes, total_events, embed_size))
            embed_size = 2 * embed_size + 1
        self.layers = ModuleList(layers)

    def forward(self, u: Tensor, v: Tensor, t: Tensor) -> Tensor:
        self.tick += 1
        u = u[:self.tick]
        v = v[:self.tick]
        t = t[:self.tick]
        g = t[-1] - t
        # no node-level embedding
        h = torch.zeros(self.total_nodes, 0)

        for layer in self.layers:
            # NB detach()!!
            layer.save(
                h[u[-1]].detach(),
                h[v[-1]].detach(),
                self.tick - 1
            )
            h = layer(u, v, g, h, self.tick)
        return h

if __name__ == '__main__':
    torch.manual_seed(0)
    dataset = PyGLinkPropPredDataset(name="tgbl-wiki", root="datasets")
    total_nodes = max(int(dataset.src.max()), int(dataset.dst.max().item())) + 1
    total_events = len(dataset.ts)
    model = T1(total_nodes, total_events)
    for event in range(1, total_events + 1):
        print(event)
        h = model(dataset.src, dataset.dst, dataset.ts)
