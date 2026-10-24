"""Слои MyGCNConv и MySAGEConv, реализованные через scatter-операции."""
import torch
from torch import nn


def scatter_sum(src, index, dim_size):
    """Сумма строк src по индексам index вдоль нулевой оси (scatter_add)."""
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, index, src)


def scatter_mean(src, index, dim_size):
    """Среднее строк src по индексам index; у пустых групп получается 0."""
    total = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones(index.size(0), dtype=src.dtype, device=src.device), index, dim_size)
    count = count.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))
    return total / count


def add_remaining_self_loops(edge_index, num_nodes):
    """Добавляет петлю каждой вершине; уже существующие петли не дублируются."""
    keep = edge_index[0] != edge_index[1]
    loop = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index[:, keep], torch.stack([loop, loop])], dim=1)


class MyGCNConv(nn.Module):
    """GCN (Kipf & Welling, 2017): h_v' = sum_{u in N(v)+v} W h_u / sqrt(d_u d_v)."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)  # glorot, как в GCNConv
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = add_remaining_self_loops(edge_index, num_nodes)

        # степень с петлёй по целевой вершине: сообщения идут src -> dst
        deg = scatter_sum(x.new_ones(dst.size(0)), dst, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        h = self.lin(x)
        out = scatter_sum(h[src] * norm.unsqueeze(-1), dst, num_nodes)
        if self.bias is not None:
            out = out + self.bias
        return out


class MySAGEConv(nn.Module):
    """GraphSAGE-mean: h_v' = W_neigh mean_{u in N(v)} h_u + W_self h_v, без петель."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_neigh.reset_parameters()
        self.lin_self.reset_parameters()
        if self.lin_neigh.bias is not None:
            nn.init.zeros_(self.lin_neigh.bias)

    def forward(self, x, edge_index):
        src, dst = edge_index
        neigh = scatter_mean(x[src], dst, x.size(0))
        return self.lin_neigh(neigh) + self.lin_self(x)
