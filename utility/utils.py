import numpy as np
import json
import os

import torch
import yaml
import os.path as osp
from utility.globals import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



def load_data():
    fts_users = np.load(args_global.data_prefix + args_global.dataset + '/node_feats.npy')
    fts_news = np.load(args_global.data_prefix + args_global.dataset + '/edge_feats.npy')
    H = np.load(args_global.data_prefix + args_global.dataset + '/hypergraph.npy')
    labels = json.load(open(args_global.data_prefix + args_global.dataset + '/edge_labels.json'))
    role = json.load(open(args.path + args.dataset + '/role' + args.fold + '.json'))
    idx_train = np.array(role['tr'])
    idx_val = np.array(role['va'])
    idx_test = np.array(role['te'])
    lbls = []
    for ind,item in enumerate(labels.items()):
        assert ind==int(item[0])
        lbls.append(item[1])
    lbls = np.array(lbls)

    return fts_users, fts_news, H, lbls, idx_train, idx_val, idx_test



def get_config():
    with open(args_global.train_config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg



def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    from 'Hypergraph Neural Networks': https://arxiv.org/abs/1809.09401
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G



def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    from 'Hypergraph Neural Networks': https://arxiv.org/abs/1809.09401
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()



def involved_users(hypergraph, idx):
    involved_dict = {}
    num = hypergraph.shape[1]
    for col in idx:
        if int(col) < num:
            col = int(col)
            inv_u = hypergraph[:,col]
            index_u = [i for i, u in enumerate(inv_u) if u==1]
            if index_u != []:
                involved_dict[col] = index_u

    return involved_dict


def Average(users_of_each_news, user_embedding, num_news):
    user_embedding = user_embedding.cpu()
    agg_embs = torch.zeros(num_news, user_embedding.shape[1])
    for new_idx, users_idx in users_of_each_news.items():
        curr_embs = [user_embedding[idx] for idx in users_idx]
        final_u_emb = sum([curr_embs[i] for i in range(len(curr_embs))])
        final_u_emb = final_u_emb / len(curr_embs)
        agg_embs[new_idx] = final_u_emb
    agg_embs = F.normalize(agg_embs)
    agg_embs = agg_embs.clone().detach().requires_grad_(True)
    return agg_embs


def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()



##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))