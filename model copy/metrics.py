from model.utils import quaternion2rot
import numpy as np
import torch
import random

from liegroups.torch.so3 import SO3Matrix as so3m

torchpi = torch.acos(torch.zeros(1)).item() * 2


def R2w(R):
    w = torch.stack((R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0]- R[0,1])) / 2
    s = torch.norm(w)
    if s:
        w = w / s * torch.atan2(s,(torch.trace(R) - 1) / 2)
    return w

def btrace(R):
    return R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).view(-1,1)

def bR2w(R):
    batchlize = False
    if len(R.shape)<3:
        R = R.unsqueeze(0)
        batchlize = True
    w = torch.stack([R[:,2,1] - R[:,1,2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=1) / 2.
    s = torch.norm(w, dim=1, keepdim=True)
    count_zero = s != 0.
    R_trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).view(-1,1)
    w[count_zero] = w[count_zero] / s[count_zero] * torch.atan2(s[count_zero], (R_trace - 1.) / 2.)
    if batchlize:
        w = w.squeeze(0)
    return w

def w2R(w):
    device = w.device
    omega = torch.norm(w)
    if omega:
        n = w / omega
        s = torch.sin(omega)
        c = torch.cos(omega)
        cc = 1- c
        n1 = n[0]
        n2 = n[1]
        n3 = n[2]
        n12cc=n1*n2*cc
        n23cc=n2*n3*cc
        n31cc=n3*n1*cc
        n1s=n1*s              
        n2s=n2*s              
        n3s=n3*s
        R = torch.zeros(3,3)
        R[0,0]=c+n1*n1*cc
        R[0,1]=n12cc-n3s      
        R[0,2]=n31cc+n2s
        R[1,0]=n12cc+n3s     
        R[1,1]=c+n2*n2*cc     
        R[1,2]=n23cc-n1s
        R[2,0]=n31cc-n2s      
        R[2,1]=n23cc+n1s      
        R[2,2]=c+n3*n3*cc
    else:
        R = torch.eye(3)
    R = R.to(device)
    return R


def compare_rot_graph_iter(R1, R2, method="median", run_time=4, count_time=20):
    sigma2 = (5. * torchpi / 180.) * (5. * torchpi / 180.)
    N = R1.shape[0]
    device = R1.device
    Emeanbest = float("Inf")
    E = torch.zeros(3).to(device)
    Ebest = E.clone()
    e = torch.zeros(N,1).to(device)
    ebest = e
    ori_R2 = R2.clone()
    new_R2 = None
    if run_time > N:
        run_time = N
    l = torch.randperm(N)[:run_time]
    delta_R1 = so3m.exp(torch.tensor([0.,0.,0.]).to(device))
    delta_R2 = so3m.exp(torch.tensor([0.,0.,0.]).to(device))
    # l = [39,38,1,93]
    for i in range(run_time):
        j = l[i]
        R = R1[j,:,:].clone().t().view(1,3,3).repeat(N,1,1)
        R1 = torch.bmm(R1, R)
        delta_R1 = delta_R1.dot(so3m.from_matrix(R[0], normalize=True))

        R = R2[j,:,:].clone().t().view(1,3,3).repeat(N,1,1)
        R2 = torch.bmm(R2, R)
        delta_R2 = delta_R2.dot(so3m.from_matrix(R[0], normalize=True))

        W = torch.zeros(N,3)
        d = float("Inf")
        count = 1
        while(d > 1e-5 and count < count_time):
            W = so3m.from_matrix(R1, normalize=True).dot(so3m.from_matrix(R2, normalize=True).inv()).to_rpy()
            # for k in range(N):
            #     W2[k,:] = R2w(torch.mm(R1[k,:,:],R2[k,:,:].t()))
            
            if method == "mean":
                w = torch.mean(W, 0)
                d = torch.norm(w)
                R = so3m.from_rpy(w).as_matrix()
            elif method == "median":
                w = torch.median(W, 0).values
                d = torch.norm(w)
                R = so3m.from_rpy(w).as_matrix()
            elif method == "robustmean":
                w = 1. / torch.sqrt( torch.sum(W*W, 1) + sigma2)
                w = w / torch.sum(w)
                w = torch.mean(w.repeat(1,3) * W)
                d = torch.norm(w)
                R = so3m.from_rpy(w).as_matrix()
            delta_R2 = delta_R2.dot(so3m.from_matrix(R, normalize=True))
            R = R.view(-1,3,3).repeat(N,1,1)
            R2 = torch.bmm(R2, R)
            count = count + 1
        e = torch.acos(torch.clip((btrace(torch.bmm(R1, 
                                                so3m.from_matrix(R2, normalize=True).inv().as_matrix()
                                                )) - 1. ) / 2., min=-1.0, max=1.0))

        e = e * 180.0 / torchpi
        E = torch.stack([torch.mean(e),torch.median(e), torch.sqrt(torch.mm(e.t(),e)/len(e))[0,0]])
        if E[1] < Emeanbest:
            ebest= e
            Ebest = E
            Emeanbest = E[2]
            new_R2 = (R1, R2)
    return Ebest,  new_R2


def compare_rot_graph_SVD(R1, R2):
    if R1.shape[-1] == 4:
        R1 = quaternion2rot(R1)
        R2 = quaternion2rot(R2)
    
    if len(R1.shape) < 3:
        R1 = R1.unsqueeze(0)
        R2 = R2.unsqueeze(0)

    A = torch.bmm(R1.transpose(-1,-2), R2).sum(dim=0)
    u,s,v = torch.svd(A)
    new_s = torch.eye(3).to(R1.device) 
    new_s[2,2] = torch.det(torch.mm(u,v.T))
    R_align = torch.mm(torch.mm(u,new_s), v.T)

    R1_align = torch.bmm(R1, R_align.view(1,3,3).repeat(R1.shape[0],1,1))
    e = torch.acos(torch.clip((btrace(torch.bmm(R1_align, 
                                                R2)) - 1. ) / 2., min=-1.0, max=1.0))
    e = e * 180.0 / torchpi
    E = torch.stack([torch.mean(e),torch.median(e), torch.sqrt(torch.mm(e.t(),e)/len(e))[0,0]])
    return E, R_align

def read_rot(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    lines = [x.split()[1:] for x in lines]
    lines = [ [float(y) for y in x] for x in lines]
    R = np.asarray(lines)
    R = R.reshape((-1,3,3))
    return R
if __name__ == "__main__":
    file1 = "./data/R1.txt"
    file2 = "./data/R2.txt"
    R1 = read_rot(file1)
    R2 = read_rot(file2)
    R1 = torch.from_numpy(R1).to(torch.float64)
    R2 = torch.from_numpy(R2).to(torch.float64)
    e = compare_rot_graph_iter(R1,R2)
    print(e)