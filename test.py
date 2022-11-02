"""Train the model"""

from model.utils import quaternion2rot

import torch

from tqdm import tqdm
from easydict import EasyDict as edict
import model.utils as utils
from model.rago_model import rago as net
from model.loss import compute_losses
from time import time
from model.metrics import compare_rot_graph_iter

def test(model, test_ds):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0.0
    total_rot_err = []
    with torch.no_grad():
        with tqdm(total=len(test_ds)) as t:
            for i, data_batch in enumerate(test_ds):
                # move to GPU if available
                data_batch = data_batch.cuda() 
                init_rot = None
                data_batch.edge_attr = quaternion2rot(data_batch.edge_attr)
                data_batch.gt_rot = quaternion2rot(data_batch.gt_rot)

                output_batch = model(data_batch, init_rot, 8, 1, 4)
                loss = compute_losses(data_batch, output_batch)
                gt_rot = data_batch.gt_rot
                for i,rot in enumerate(output_batch["est_nodes"]):
                    aligned_rotation_err = utils.compare_rot_graph_iter(rot.view(-1,3,3), gt_rot.view(-1,3,3))[0]
                    if len(total_rot_err) <= i:
                        total_rot_err.append(aligned_rotation_err)
                    else:
                        total_rot_err += (aligned_rotation_err)
                
                total_loss += loss["total"]"""Train the model"""

from model.utils import quaternion2rot
import torch

import test
import torch.optim as optim
from tqdm import tqdm
import model.utils as utils
from model.rago_model import rago as net
from model.loss import compute_losses

def train(model, train_ds, optimizer, scheduler, epoch):
    torch.cuda.empty_cache()
    model.train()

    with tqdm(total=len(train_ds)) as t:
        for i, data_batch in enumerate(train_ds):
            # move to GPU if available
            data_batch = data_batch.cuda() 
            if data_batch.edge_attr.shape[0] > 600000:
                t.update()
                continue
            data_batch = utils.random_sample_subgraph_from_data_batch_rotation(data_batch, ratio=0.8)
            init_rot = None
            data_batch.edge_attr = quaternion2rot(data_batch.edge_attr)
            data_batch.gt_rot = quaternion2rot(data_batch.gt_rot)

            output_batch = model(data_batch, init_rot, 3, 1, 4)
            loss = compute_losses(data_batch, output_batch)

            optimizer.zero_grad()
            loss["total"].backward()
            optimizer.step()
            t.update()

    if  epoch > 100:
        scheduler.step()

def train_and_evaluate(model, train_ds, eval_ds,test_ds, optimizer, scheduler):
    for epoch in range(1000000):
        train(model, train_ds, optimizer, scheduler)
        tloss = eval(model, eval_ds)
        # save model
        if epoch % 100 == 0:
            test(model, test_ds)

def main(config=None):
    model = net().cuda()

    train_ds, eval_ds,test_ds = None, None,None

    # add regulizer to weights and bias
    param_groups = [
        {"params": utils.bias_parameters(model)},
        {"params": utils.weight_parameters(model), "weight_decay": 1e-6},
    ]
    
    optimizer = optim.AdamW(param_groups, lr=0.0001,
                                                   betas=(0.9, 0.999), eps=1e-7)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    train_and_evaluate(model, train_ds, eval_ds,test_ds, optimizer, scheduler)


if __name__ == '__main__':

    main()

                t.update()
    for i,err in enumerate(total_rot_err):
        print(".2f .2f" % (err[0],err[1]))

    return total_loss

def eval(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status
    for k, v in manager.loss_status.items():
        manager.loss_status[k].reset()
    for k,v in manager.val_status.items():
        manager.val_status[k] = 99999
    # set model to training mode


    torch.cuda.empty_cache()
    model.eval()
    cnt = 0
    total_error = None
    total_error_seq = []
    total_time = 0
    # Use tqdm for progress bar
    with torch.no_grad():
        with tqdm(total=len(manager.val_dataloader)) as t:
            for i, data_batch in enumerate(manager.val_dataloader):
                cnt +=1

                # move to GPU if available
                data_batch = data_batch.cuda()
                init_rot = None
                    
                data_batch.edge_attr = quaternion2rot(data_batch.edge_attr)
                data_batch.gt_rot = quaternion2rot(data_batch.gt_rot)
                manager.train_status['print_str'] = 'Eval: '
                # compute model output and loss
              
                output_batch = model(data_batch, 6, fix_edge=False,init_rot=init_rot)
                loss = compute_losses(data_batch, output_batch, manager)

                gt_quat = data_batch.gt_rot
                gt_id = data_batch.gt_id
                edge_attr = data_batch.edge_attr.view(-1,3,3)
                edge_index = data_batch.edge_index

                rot_seq = []
                for quat in output_batch["w_on_nodes"]:
                    aligned_rotation_err = compare_rot_graph_iter((quat[gt_id]).view(-1,3,3), (gt_quat[gt_id]).view(-1,3,3))[0]
                    rot_seq.append(aligned_rotation_err)

                i=-1
                min_eval_loss = 9999999
                now_error = None
                for quat in output_batch["w_on_nodes"]:
                    i+=1
                    eval_now_loss = loss["edge_seq_loss"][i]
                    if eval_now_loss < min_eval_loss:
                        min_eval_loss = eval_now_loss
                        if now_error is None:
                            now_error = rot_seq[i]
                        else :
                            now_error += rot_seq[i]

                if total_error is None:
                    total_error = now_error
                else :
                    total_error += now_error

                for i,aligned_rot in enumerate(rot_seq):
                    if len(total_error_seq) < i+1:
                        total_error_seq.append(aligned_rot.clone().detach().cpu())
                    else:
                        total_error_seq[i] += aligned_rot.clone().detach().cpu()
                

                t.set_description(desc=manager.train_status['print_str'])
                t.update()
    print("time: %.8f s/iter" % (total_time / cnt))
    for i, rot_err in enumerate(total_error_seq):
        print(i, "   ", rot_err / cnt)
    
