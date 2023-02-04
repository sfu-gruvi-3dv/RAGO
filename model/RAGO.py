import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch_scatter import scatter
from liegroups.torch.so3 import SO3Matrix as so3m
from model.base_model import MLP, MLPGRU, MPNN_1Conv, MPNN_3Conv
import model.utils as utils
class rago(nn.Module):
    def __init__(self):
        
        self.mpnn_feat = MPNN_1Conv(9,9,48)
        self.mpnn_state = MPNN_1Conv(9,9,48)
        
        self.mpnn_edge_cost = MPNN_3Conv(9,9,48)
        self.mpnn_node_cost = MPNN_3Conv(9,9,48)
        
        self.edge_fusion = MLP(6,[3],48+48, 48, 48)
        self.edge_gru = MLPGRU(48+48, 48)
        self.edge_updater = MLP(6,[3], 48,  48, 6)
        
        self.node_fusion = MLP(6,[3],48+48, 48, 48)
        self.node_gru = MLPGRU(48+48, 48)
        self.node_updater = MLP(6,[3], 48,  48, 6)

    def init(self, init_rot, cn, edge_index, edge_attr, device):
        if init_rot is None:
            camera_rotation = torch.rand(cn,4).to(device)
            camera_rotation[:, 1:] -= 0.5
            camera_rotation = F.normalize(camera_rotation, dim=1, p=2)
            camera_rotation = utils.quaternion2rot(camera_rotation)
            camera_rotation = camera_rotation.detach()
            init_rot = camera_rotation.detach()
        rect_edges = torch.zeros(edge_attr.shape[0],6).to(device)

        node_feat, edge_feat = self.mpnn_feat(init_rot, edge_index, edge_attr)
        node_state, edge_state = self.mpnn_state(init_rot, edge_index, edge_attr)
        node_state, edge_state = torch.tanh(node_state), torch.tanh(edge_state)
        
        return init_rot, rect_edges, node_feat, edge_feat, node_state, edge_state
    
    def compute_graph_cost(self, node_rotations, edge_rotations, edge_attr, edge_index):
        # sra node cost
        edge_start = node_rotations[edge_index[0]]
        edge_end = node_rotations[edge_index[1]]
        edge_start_end = torch.bmm(edge_attr.view(-1,3,3).transpose(1,2), edge_end.view(-1,3,3))
        
        sra_cost = scatter((edge_start_end - edge_end).abs(), edge_index[1], dim=0, 
                           dim_size=node_rotations.shape[0], reduce="mean")
        
        # edge cost
        edge_rot_mat = utils.compute_rotation_matrix_from_ortho6d(edge_rotations)
        node_rel = utils.edge_model_rot(node_rotations, edge_index)
        edge_cost = (edge_rot_mat - edge_attr).abs() + (edge_rot_mat - node_rel).abs()

        return sra_cost, edge_cost
    
    def forward(self, data_batch, init_rot=None, tg=3, te=1, tn=4):
        edge_attr = data_batch.edge_attr
        edge_index = data_batch.edge_index
        device = edge_attr.device
        
        # 3*3, 6, 48,48,48,48,48
        node_rotations, edge_rotations, node_feat, edge_feat, node_state, edge_state = \
            self.init(init_rot, data_batch.x.shape[0],edge_index, edge_attr,device)
        
        ret = {}
        ret["rect_edges"] = []
        ret["est_nodes"] = []
        
        for ig in range(tg):
            for ie in range(te):
                
                node_rotations = node_rotations.detach()
                edge_rotations = edge_rotations.detach()
                
                node_input = node_rotations.detach().view(-1, 9)
                edge_input = edge_rotations.detach().view(-1, 9)
                node_cost, edge_cost = self.compute_graph_cost(node_rotations, edge_rotations,
                                                      edge_attr, edge_index)
                _, edge_cost_feat = self.mpnn_edge_cost(node_cost, edge_index, edge_cost)
                edge_fusion_feat = self.edge_fusion(torch.cat([edge_cost_feat, edge_feat], dim=-1))
                edge_fusion_feat = torch.cat([edge_fusion_feat, edge_feat], dim=-1)
                
                edge_state = self.edge_gru(edge_state, edge_fusion_feat)
                edge_state = torch.tanh(edge_state)
                
                delta_edge_refine = self.edge_updater(edge_state)
                edge_rotations += delta_edge_refine
                ret["rect_edges"].append(edge_rotations)
        
            for ino in range(tn):
                node_rotations = node_rotations.detach()
                edge_rotations = edge_rotations.detach()
                
                node_input = node_rotations.detach().view(-1, 9)
                edge_input = edge_rotations.detach().view(-1, 9)
                node_cost, edge_cost = self.compute_graph_cost(node_rotations, edge_rotations,
                                                      edge_attr, edge_index)
                node_cost_feat,_ = self.mpnn_node_cost(node_cost, edge_index, edge_cost)
                node_fusion_feat = self.node_fusion(torch.cat([node_cost_feat, node_feat], dim=-1))
                node_fusion_feat = torch.cat([node_fusion_feat, node_feat], dim=-1)
                
                node_state = self.node_gru(node_state, node_fusion_feat)
                node_state = torch.tanh(node_state)
                
                delta_node_rotations = self.node_updater(node_state)
                delta_node_rotations = utils.compute_rotation_matrix_from_ortho6d(delta_node_rotations)
                node_rotations = torch.bmm(node_rotations.view(-1,3,3), delta_node_rotations.view(-1,3,3))
                camera_rotation = so3m.from_matrix(camera_rotation, normalize=True).as_matrix().view(-1,9)
                
                ret["est_nodes"].append(camera_rotation)
        return ret
