import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class StarGAT_nc(nn.Module):
    def __init__(self,
                 feats_dim_list,
                 hidden_dim,
                 num_heads,
                 out_dim,        
                 attn_vec_dim,   
                 target_node_num,
                 dropout_rate=0.5):
        super(StarGAT_nc, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.target_node_num = target_node_num

        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x

        self.dropout_rate = dropout_rate
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # StarGAT_nc layers
        self.layer = StarGAT_nc_layer(hidden_dim, hidden_dim, attn_vec_dim, target_node_num, attn_drop=dropout_rate)

        # gamma GAT
        self.fc1 = nn.Linear(hidden_dim, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

            
    def forward(self, inputs):
    
        g_list, features_list, type_mask, edge_list, target_idx_list, adjM, idx_batch = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i]) 

        h = self.feat_drop(transformed_features)

        h, _ = self.layer((g_list, h, type_mask, edge_list, target_idx_list))
        h = F.elu(h)        

        adjM = adjM.cpu()        
        emb = []
        emb.append(torch.unsqueeze(h, 1))
        idx = np.where(adjM[idx_batch] > 0)[1]
        emb.append(transformed_features[idx].view(len(idx_batch), -1, self.hidden_dim))
        emb = torch.cat(emb, dim=1)
        fc1 = torch.tanh(self.fc1(emb))
        fc2 = self.fc2(fc1)
        
        gamma = F.softmax(fc2, dim=1)
        h_prime = torch.sum(gamma * emb, dim=1)
        
        return self.fc_final(h_prime), h_prime

class StarGAT_nc_layer(nn.Module):
    def __init__(self,
                 in_dim,      
                 out_dim,     
                 attn_vec_dim, 
                 target_node_num, 
                 attn_drop=0.5):
        super(StarGAT_nc_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.target_node_num = target_node_num
            
        self.ctr_ntype_layer = StarGAT_ctr_ntype_specific(in_dim, attn_vec_dim, target_node_num, attn_drop)

        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        h = self.ctr_ntype_layer(inputs)
        h_fc = self.fc(h) 
        return h_fc, h
        
class StarGAT_ctr_ntype_specific(nn.Module):
    def __init__(self, 
                 out_dim,         
                 attn_vec_dim,    
                 target_node_num,
                 attn_drop=0.5):
        super(StarGAT_ctr_ntype_specific, self).__init__()
        self.num_metapaths = 2 
        self.out_dim = out_dim
        self.target_node_num = target_node_num

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(self.num_metapaths):
            self.metapath_layers.append(StarGAT_metapath_specific(out_dim, attn_drop=attn_drop))

        self.fc1 = nn.Linear(out_dim, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):

        g_list, features, type_mask, edge_list, target_idx_list = inputs

        # metapath-specific layers
        metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge, target_idx)).view(-1, self.out_dim))
                         for g, edge, target_idx, metapath_layer in zip(g_list, edge_list, target_idx_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)

        h = torch.sum(beta * metapath_outs, dim=0)
        return h

class StarGAT_metapath_specific(nn.Module):
    def __init__(self,
                 out_dim,
                 attn_drop=0.5,
                 alpha=0.01):
        super(StarGAT_metapath_specific, self).__init__()
        self.out_dim = out_dim

        self.attn = nn.Parameter(torch.empty(size=(1, out_dim)))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        g, features, type_mask, edge, target_idx = inputs
        edata = F.embedding(edge, features)
        eft = torch.mean(edata, dim=1)   
 
        a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        self.edge_softmax(g)
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']

        return ret[target_idx]

