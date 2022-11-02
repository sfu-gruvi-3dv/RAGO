import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import *
class QuaternionSPT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, edge_attr, edge_index, gt_rot=None, gt_id=None, start_node=None):
        # The edge list MUST contain ONLY ONE Connect component
        # The node doesn't show in the edge index will be setted to [1,0,0,0]
        # The node index start from 0
        # edge_attr = [M,4]
        # edge_index = [2,M]
        # gt_id = [M] BOOL
        # start_node = int
        device = edge_attr.device

        max_node = edge_index.max().long()  
        node_num = max_node + 1
        node_list = torch.arange(0,node_num, step=1).long().to(device)
        
        if gt_id is None:
            gt_id = torch.ones(node_num).bool().to(device)
            gt_node_num = node_num
        else:
            gt_node_num = gt_id.sum()
        if start_node is None:
            start_node = node_list[gt_id][torch.randperm(gt_node_num)[0]]

        # start generate spanning tree from start_node
        edge_rel_degree = quaternion2deg(edge_attr)

        # rebuild edge list, based on node index
        edge_list = dict()
        edge_err_list = dict()

        for i in range(node_num):
            is_u = (edge_index.t()[:,0] == i).view(-1)
            edge_err = edge_rel_degree[is_u].view(-1)
            edge_att = edge_attr[is_u].view(-1, 4)
            edge_ind = edge_index.t()[:,1][is_u].view(-1,1).float()
            edge_list[i] = torch.cat([edge_ind, edge_att], dim=1)
            edge_err_list[i] = edge_err

        # sorting edge list for each node by relative error

        new_edge_list = dict()
        for k,v in edge_err_list.items():
            nv, vi = torch.sort(v, dim=0)
            el = edge_list[k]
            new_edge_list[k] = el[vi]

        visited = torch.zeros(node_num).bool().to(device)
        visited[start_node] = True
        que = []
        que.append(start_node)

        node_quat = torch.zeros(node_num,4)
        node_quat[:,0] = 1.
        node_quat = node_quat.to(device)
        

        
        while(len(que)):
            u = que[0]
            que.pop(0)

            for edge in new_edge_list[u]:
                v = edge[0].long().item()
                rel_quat = edge[1:5]
                rel_err = edge[5:]
                if visited[v] == False:
                    visited[v] = True
                    que.append(v)
                    node_quat[v,:] = qmul(inv_q(rel_quat), node_quat[u])

        return start_node, node_quat
        


if __name__ == "__main__":
    node_num = 600
    edge_attr = F.normalize(torch.rand(80000,4), p=2, dim=1)
    edge_index = (torch.rand(2,80000) * node_num).long()
    start_node = 120

    gg = QuaternionSPT()
    node, spt = gg(edge_attr, edge_index, start_node=120)
    print(node)