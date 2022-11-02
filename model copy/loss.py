import torch


from model import utils as utils
def seq_edge_loss(edge_rotations,edge_attr, gt_rot, edge_index, gamma=0.8):
    loss = 0
    seq_num = len(edge_rotations)
    losses = []
    gt_edge_quat = utils.rel_rot_from_global(gt_rot, edge_index)
    gt_edge_quat_inv = gt_edge_quat.transpose(1,2)
    gt_gap_rot = torch.bmm(gt_edge_quat_inv, edge_attr)
    for i, pred_rot in enumerate(edge_rotations):
        now_coff = gamma ** (seq_num - i - 1)
        edge_refine = utils.compute_rotation_matrix_from_ortho6d(pred_rot)
        edge_loss = (edge_refine - gt_gap_rot).abs()
        edge_loss = (edge_loss).mean()
        loss += now_coff * edge_loss
    return loss

def seq_node_loss(node_rotations, edge_attr, edge_index, gt_rot, gamma=0.8):
    loss = 0
    seqs = []

    seq_num = len(node_rotations)
    gt_rel_rot = utils.rel_rot_from_global(gt_rot, edge_index)
    gt_rel_rot_inv = gt_rel_rot.transpose(1,2)
    for i, pred_rot in enumerate(node_rotations):
        now_coff = (gamma) ** (seq_num - 1 - i)
        pred_rel_rot = utils.rel_rot_from_global(pred_rot.view(-1,3,3), edge_index)
        edge_loss =  (pred_rel_rot - gt_rel_rot).abs()
        edge_loss = (edge_loss.mean()) * now_coff
        loss += edge_loss
    return loss

def compute_losses(data_batch, output_batch):
    gt_rot = data_batch.gt_rot
    edge_rotations = output_batch["rect_edges"]
    node_rotations = output_batch["est_nodes"]
    
    edge_attr = data_batch.edge_attr.view(-1,3,3)
    edge_index = data_batch.edge_index
    loss = {}
    loss["edge"] = seq_edge_loss(edge_rotations, edge_attr, gt_rot, edge_index)
    loss["node"] = seq_node_loss(node_rotations, edge_attr, edge_index, gt_rot)
    loss["total"] = loss["edge"] + loss["node"]
    return loss
