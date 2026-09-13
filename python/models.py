
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, feature_dtype):
        super().__init__()
        self.layers = nn.ModuleList()
        if(num_layers == 1):
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.convert_dtype = feature_dtype != torch.float32

    def forward(self, blocks, x):
        hidden_x = x
        if(self.convert_dtype):
            hidden_x = hidden_x.to(dtype=torch.float32)
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


class GCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, feature_dtype):
        super().__init__()
        self.layers = nn.ModuleList()
        if(num_layers == 1):
            self.layers.append(dglnn.GraphConv(in_size, out_size, allow_zero_in_degree=True))
        else:
            self.layers.append(dglnn.GraphConv(in_size, hidden_size, activation=F.relu, allow_zero_in_degree=True))
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hidden_size, hidden_size, activation=F.relu, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hidden_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)
        self.convert_dtype = feature_dtype != torch.float32

    def forward(self, blocks, features):
        h = features
        if(self.convert_dtype):
            h = h.to(dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h


class GAT(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, feature_dtype, heads=8):
        super().__init__()
        try:
            len(heads)
        except TypeError:
            heads = [heads] * num_layers
        self.layers = nn.ModuleList()
        if(num_layers == 1):
            self.layers.append(
                dglnn.GATConv(
                    in_size,
                    out_size,
                    heads[0],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=None,
                    allow_zero_in_degree=True
                )
            )
        else:
            self.layers.append(
                dglnn.GATConv(
                    in_size,
                    hidden_size,
                    heads[0],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
            )
            for i in range(num_layers - 2):
                self.layers.append(
                    dglnn.GATConv(
                        hidden_size * heads[i],
                        hidden_size,
                        heads[i+1],
                        feat_drop=0.6,
                        attn_drop=0.6,
                        activation=F.elu,
                        allow_zero_in_degree=True
                    )
                )
            self.layers.append(
                dglnn.GATConv(
                    hidden_size * heads[num_layers-2],
                    out_size,
                    heads[num_layers-1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=None,
                    allow_zero_in_degree=True
                )
        )
        #self.dropout = nn.Dropout(0.5) # seems already in the layers
        self.convert_dtype = feature_dtype != torch.float32

    def forward(self, blocks, features):
        h = features
        if(self.convert_dtype):
            h = h.to(dtype=torch.float32)
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            is_last_layer = layer_idx == len(self.layers) - 1
            #if not is_last_layer:
            #    h = self.activation(h)
            #    h = self.dropout(h)
            if(is_last_layer):
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
