# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .conv_layers import GINConv, GINEConv, SGCConv

import torch

class SGC(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        self.hidden_size = hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(SGCConv(hidden_size, hidden_size))
            else:
                self.convs.append(SGCConv(hidden_size, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size,elementwise_affine=True))

        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None,bsize=-1,last_att=False,att_opt='all',return_data='pred'):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)
        if att_opt=='mean':
            final_x = []
            for sel_i in range(self.n_layers):
                h = x.clone()
                for i in range(self.n_layers):
                    if i!=sel_i:
                        h = self.convs[i](h, edge_index, edge_attr=edge_attr)
                    else:
                        h = self.convs[i](h, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
                    h = self.relu(h)
                    h = F.dropout(h, p=self.dropout_p, training=self.training)
                if bsize > 0:
                    h = self.pool(h,batch,bsize)
                else:
                    h = self.pool(h, batch)
                h = self.fc_out(h)
                # print(h.mean())
                final_x.append(h)
            x = torch.stack(final_x)#.mean(dim=0)
        elif att_opt=='first':
            for i in range(self.n_layers):
                if i!=0:
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr)
                else:
                    # print(f"gogogo {i}")
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
                x = self.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            if bsize > 0:
                h = self.pool(x,batch,bsize)
            else:
                h = self.pool(x, batch)
            x = self.fc_out(h)
        elif att_opt=='norm':
            for i in range(self.n_layers):
                if False:
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr)
                else:
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
                x = self.norms[i](x)
                x = self.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            if bsize > 0:
                h = self.pool(x,batch,bsize)
            else:
                h = self.pool(x, batch)
            x = self.fc_out(h)
        else:
            for i in range(self.n_layers):
                if last_att and i+1<self.n_layers:
                # if last_att and i==0:
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr)
                else:
                    # print(f"gogogo {i}")
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
                x = self.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            if bsize > 0:
                h = self.pool(x,batch,bsize)
            else:
                h = self.pool(x, batch)
            x = self.fc_out(h)
        if return_data == 'feats':
            return x, h
        return x
    # def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
    #     x = self.node_encoder(x)
    #     if edge_attr is not None and self.use_edge_attr:
    #         edge_attr = self.edge_encoder(edge_attr)

    #     for i in range(self.n_layers):
    #         x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
    #         x = self.relu(x)
    #         x = F.dropout(x, p=self.dropout_p, training=self.training)
    #     return self.fc_out(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
