"""Train the model"""

import argparse
from model.utils import quaternion2rot
import os
import torch
import torchvision as tv
import json
import datetime
import random

import test
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from easydict import EasyDict as edict
import model.utils as utils
from model.RAGO import rago as net
from model.loss import compute_losses
from time import time

def eval(model, eval_ds):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(total=len(eval_ds)) as t:
            for i, data_batch in enumerate(eval_ds):
                # move to GPU if available
                data_batch = data_batch.cuda() 
                init_rot = None
                data_batch.edge_attr = quaternion2rot(data_batch.edge_attr)
                data_batch.gt_rot = quaternion2rot(data_batch.gt_rot)

                output_batch = model(data_batch, init_rot, 3, 1, 4)
                loss = compute_losses(data_batch, output_batch)
                total_loss += loss["total"]
                t.update()

    return total_loss
