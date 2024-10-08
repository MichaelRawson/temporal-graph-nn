import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Module, ModuleList, Linear, Parameter
from torch.nn.modules import BatchNorm1d

from hyper import EMBED, LAYERS, HIDDEN, DEVICE

class LinkPrediction(Module):
    """link prediction head"""

    hidden: Linear
    """hidden layer"""
    output: Linear
    """output layer"""

    def __init__(self, embedding_size: int):
        super().__init__()
        self.hidden = Linear(2 * embedding_size, HIDDEN)
        self.output = Linear(HIDDEN, 1)

    def forward(self, h_u: Tensor, h_v: Tensor) -> Tensor:
        h = torch.cat((h_u, h_v), 1)
        return self.output(torch.relu(self.hidden(h))).squeeze()


class T12(Module):
    """base class for T1 and T2"""

    agg_features: int
    """number of features after aggregation"""
    bn: BatchNorm1d
    """batch normalisation"""
    zeros: Tensor
    """appropriately-sized zero buffer for scatter ops"""
    w1: Linear
    """first linear transform"""
    w2: Linear
    """second linear transform"""

    def __init__(self, total_nodes: int, total_events: int, previous_embed_size: int):
        super().__init__()
        _ = total_events
        self.agg_features = previous_embed_size + 1
        self.register_buffer('zeros', torch.zeros(total_nodes, self.agg_features), persistent=False)
        self.bn = BatchNorm1d(self.agg_features)
        self.w1 = Linear(self.agg_features, self.agg_features)
        out_size = self.agg_features + previous_embed_size
        self.w2 = Linear(out_size, out_size)

    def after_aggregation(self, h: Tensor, agg: Tensor) -> Tensor:
        return self.w2(torch.cat((h, torch.relu(self.w1(self.bn(agg)))), 1))


class T1(T12):
    """a single layer in a T1 model"""

    remember_u: Tensor
    """(stale, non-differentiable) h_v(t') at previous layer if {u, v} is an edge in the temporal graph at t'"""
    remember_u: Tensor
    """(stale, non-differentiable) h_u(t') at previous layer if {u, v} is an edge in the temporal graph at t'"""

    def __init__(self, total_nodes: int, total_events: int, previous_embed_size: int):
        super().__init__(total_nodes, total_events, previous_embed_size)
        self.register_buffer('remember_u', torch.zeros(total_events, previous_embed_size), persistent=False)
        self.register_buffer('remember_v', torch.zeros(total_events, previous_embed_size), persistent=False)

    def remember(self, h_u: Tensor, h_v: Tensor, event: int):
        """remember h_v(t) and h_u(t) for future reference"""
        self.remember_u[event:event + h_u.shape[0]] = h_u
        self.remember_v[event:event + h_v.shape[0]] = h_v

    def forward(self, u: Tensor, v: Tensor, g: Tensor, h: Tensor, event: int) -> Tensor:
        # move dimensions around a bit, should be cheap
        u = u.unsqueeze(1).expand(-1, self.agg_features)
        v = v.unsqueeze(1).expand(-1, self.agg_features)
        remember_u = self.remember_u[:event]
        remember_v = self.remember_v[:event]

        # aggregate into v
        src_u = torch.cat((
            remember_u,
            g
        ), 1)
        agg_v = torch.scatter_add(self.zeros, 0, v, src_u)

        # aggregate into u
        src_v = torch.cat((
            remember_v,
            g
        ), 1)
        agg_u = torch.scatter_add(self.zeros, 0, u, src_v)

        agg = agg_u + agg_v
        return self.after_aggregation(h, agg)


class T2(T12):
    """a single layer in a T2 model"""

    def remember(self, *_):
        """don't need to remember anything"""
        pass

    def forward(self, u: Tensor, v: Tensor, g: Tensor, h: Tensor, _: int) -> Tensor:
        src_u = torch.cat((
            h[u],
            g
        ), 1)
        src_v = torch.cat((
            h[v],
            g
        ), 1)

        u = u.unsqueeze(1).expand(-1, self.agg_features)
        v = v.unsqueeze(1).expand(-1, self.agg_features)
        agg_u = torch.scatter_add(self.zeros, 0, u, src_v)
        agg_v = torch.scatter_add(self.zeros, 0, v, src_u)

        agg = agg_u + agg_v
        return self.after_aggregation(h, agg)


class Model(Module):
    """a T1/T2 model"""

    total_nodes: int
    """total nodes in the graph"""
    layers: ModuleList
    """the embedding layers for this model"""
    link: LinkPrediction
    """output layer"""

    def __init__(self, flavour: str, total_nodes: int, total_events: int):
        super().__init__()
        self.total_nodes = total_nodes
        Layer = {'T1': T1, 'T2': T2}[flavour]
        layers = []
        embed_size = EMBED
        for _ in range(LAYERS):
            layers.append(Layer(total_nodes, total_events, embed_size))
            embed_size = 2 * embed_size + 1

        self.layers = ModuleList(layers)
        self.link = LinkPrediction(embed_size)
        self.register_buffer('h0', torch.rand(total_nodes, EMBED), persistent=True)

    def embed(self, u: Tensor, v: Tensor, t: Tensor, event: int) -> list[Tensor]:
        """compute the embedding for each node at the present time"""

        u = u[:event]
        v = v[:event]
        t = t[:event]
        # special-case for the first event
        if event == 0:
            tfirst = 0
            tlast = 0
        else:
            tfirst = t[0]
            tlast = t[-1]

        g = ((tlast - t) / (1 + tlast - tfirst)).unsqueeze(1)
        # no node-level embedding, use random colours
        h = self.h0

        hs = [h]
        for layer in self.layers:
            h = layer(u, v, g, h, event)
            hs.append(h)
        return hs

    def remember(self, hs: list[Tensor], u: Tensor, v: Tensor, event: int):
        """remember this embedding for future reference"""
        for h, layer in zip(hs, self.layers):
            # NB detach()!!
            layer.remember(
                h[u].detach(),
                h[v].detach(),
                event
            )

    def predict_link(self, h: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """given an embedding, predict whether {u, v} at the next time point"""

        return self.link(h[u], h[v])
