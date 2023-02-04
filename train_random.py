import torch
from model.RAGO import rago as net
from easydict import EasyDict as edict
from model.loss import compute_losses
if __name__=="__main__":
    model = net().to("cuda:0")
    
    data_batch = edict()
    data_batch.device = "cuda:0"
    data_batch.edge_attr = torch.rand(100,3,3).to(data_batch.device)
    data_batch.edge_index = torch.randint(0,10,(2,100)).to(data_batch.device)
    data_batch.x = torch.rand(10,3,3).to(data_batch.device)
    data_batch.gt_rot = torch.rand(10,3,3).to(data_batch.device)
    output_batch = model(data_batch)
    print(output_batch)
    loss = compute_losses(data_batch, output_batch)
    print(loss)
    loss["total"].backward()