from abc import abstractmethod
import os
import datasets.sfminit as sfminit
from torch.utils.data import Dataset
from dataset.sfminit.sfminittypes import EG
import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import model.utils as qua_op
from liegroups.torch.so3 import SO3Matrix as so3m
import h5py
import hdf5storage
from model.quat_spt import *
from tqdm import tqdm
class graph_object:
    def __init__(self, cc, egs, bundle, valid="cc") -> None:
        self.cc = cc
        self.egs = egs
        self.bundle = bundle
        self.valid_bundle_id = set()

        # get the valid camera id in bundle
        for i, c in enumerate(self.bundle.cameras):
            rot = c.R
            if np.sum(np.abs(rot)) != 0:
                self.valid_bundle_id.add(i)

        self.valid_bundle_id = list(self.valid_bundle_id)
        self.valid = valid

        # processing valid id for egs, the new ids will stored in self.cc
        if self.valid == "all":
            # Do nothing, keep all egs as input
            new_cc = set()
            for eg in egs:
                new_cc.add(eg.i)
                new_cc.add(eg.j)
            self.cc = list(new_cc)
        elif self.valid == "cc":
            # only compute egs in cc
            new_valid_bundle_id = []
            for bid in self.valid_bundle_id:
                if bid not in self.cc:
                    continue
                new_valid_bundle_id.append(bid)
            self.valid_bundle_id = new_valid_bundle_id
        elif self.valid == "bundle":
            # only compute egs in bundle
            new_cc = []
            for c in self.cc:
                if c in self.valid_bundle_id:
                    new_cc.append(c)
            self.cc = new_cc
        
        # filter invalid eg from egs
        new_egs = []
        for eg in self.egs:
            if eg.i not in self.cc or eg.j not in self.cc:
                continue
            new_egs.append(eg)
        
        self.egs = new_egs

        # mapping id to new id

        self.map_oriid_to_ccid = dict()
        self.map_ccid_to_oriid = dict()

        for c in self.cc:
            if c not in self.map_oriid_to_ccid:
                self.map_oriid_to_ccid[c] = len(self.map_oriid_to_ccid)
                self.map_ccid_to_oriid[self.map_oriid_to_ccid[c]] = c
        
        # build mapping graph
        self.map_cc = []
        self.map_egs = []
        self.gt_rot = []
        self.gt_id = []
        for c in self.cc:
            self.map_cc.append(self.map_oriid_to_ccid[c])
        for eg in self.egs:
            self.map_egs.append(EG(self.map_oriid_to_ccid[eg.i],
                                    self.map_oriid_to_ccid[eg.j],eg.R,eg.t))
        for vid in self.cc:
            c = self.bundle.cameras[vid].R
            self.gt_rot.append(c)

        for vid in self.cc:
            self.gt_id.append(0 if vid not in self.valid_bundle_id else 1)

        # make inverse egs
        self.egs_inv = []
        for eg in self.egs:
            self.egs_inv.append(EG(eg.j, eg.i, eg.R.T, - np.matmul(eg.R.T, eg.t)))
        
        self.map_egs_inv = []
        for eg in self.map_egs:
            self.map_egs_inv.append(EG(eg.j, eg.i, eg.R.T, - np.matmul(eg.R.T, eg.t)))

        # make all edge
        self.all_egs = []
        self.all_map_egs = []
        for eg,eg_inv in zip(self.egs, self.egs_inv):
            self.all_egs.append(eg)
            self.all_egs.append(eg_inv)
        for eg, eg_inv in zip(self.map_egs, self.map_egs_inv):
            self.all_map_egs.append(eg)
            self.all_map_egs.append(eg_inv)
        
        self.all_edge_index = []
        self.all_edge_attr = []
        self.all_edge_type = []
        for i in self.all_map_egs:
            self.all_edge_index.append([i.i, i.j])
            self.all_edge_attr.append(i.R)
    
    def make_graph(self, rot_type="quat"):
        # Data, nx 
        if rot_type == "quat":
            input_node = torch.zeros([len(self.cc), 4]).float()
            gt_rot =  qua_op.rot2quaternion(torch.stack([torch.from_numpy(r) for r in self.gt_rot], dim=0)).float()
            gt_id = torch.tensor(self.gt_id).bool()
            edge_rot = qua_op.rot2quaternion(torch.stack([torch.from_numpy(r) for r in self.all_edge_attr], dim=0)).float()
            edge_id = torch.stack([torch.tensor(x) for x in self.all_edge_index], dim=0).T.long()
            # edge_rot[edge_rot[:,0]<0,:] = -edge_rot[edge_rot[:,0]<0]
            # gt_rot[gt_rot[:,0]<0,:] = -gt_rot[gt_rot[:,0]<0]
            x_i = gt_id[edge_id[0,:]]
            x_j = gt_id[edge_id[1,:]]
            valid_edge = (x_i.long() + x_j.long()) == 2
            valid_edge = valid_edge.view(edge_rot.shape[0],1)

        elif rot_type == "lieso3":
            # identity on so3
            input_node = torch.zeros([len(self.cc),3]).float()
            #log so3 
            gt_rot = so3m.from_matrix(torch.stack([torch.from_numpy(r) for r in self.gt_rot], dim=0),normalize=True).log().float()
            #
            gt_id = torch.tensor(self.gt_id)
            edge_rot = so3m.from_matrix(torch.stack([torch.from_numpy(r) for r in self.all_edge_attr], dim=0), normalize=True).log().float()
            edge_id = torch.stack([torch.tensor(x) for x in self.all_edge_index], dim=0).T.long()
            x_i = gt_id[edge_id[0,:]]
            x_j = gt_id[edge_id[1,:]]
            valid_edge = (x_i + x_j) == 2
            valid_edge = valid_edge.view(edge_rot.shape[0],1)
        elif rot_type == "rpy":
            # identity on so3
            input_node = torch.zeros([len(self.cc),3]).float()
            #log so3 
            gt_rot = so3m.from_matrix(torch.stack([torch.from_numpy(r) for r in self.gt_rot], dim=0),normalize=True).to_rpy().float()
            #
            gt_id = torch.tensor(self.gt_id)
            edge_rot = so3m.from_matrix(torch.stack([torch.from_numpy(r) for r in self.all_edge_attr], dim=0), normalize=True).to_rpy().float()
            edge_id = torch.stack([torch.tensor(x) for x in self.all_edge_index], dim=0).T.long()
            x_i = gt_id[edge_id[0,:]]
            x_j = gt_id[edge_id[1,:]]
            valid_edge = (x_i + x_j) == 2
            valid_edge = valid_edge.view(edge_rot.shape[0],1)
        
        return Data(x=input_node, gt_rot=gt_rot, gt_id=gt_id, edge_index=edge_id, edge_attr=edge_rot, valid_edge=valid_edge)

    def make_subgraph(self):
        #TODO: make subgraph
        pass

class BaseDataset(Dataset):
    def __init__(self, path=None, rot_type="qaut",on_disk=False) -> None:
        super().__init__()
        self.path = path
        self.rot_type = rot_type
        self.spt_gen = QuaternionSPT()
        self.on_disk = on_disk
        self.samples = self.collect_samples()

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self
    @abstractmethod
    def collect_samples(self):
        pass

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) :
        cache_path , i = self.samples[index]
        node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=cache_path), dtype=torch.float)
        gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)

        if self.rot_type == "rot":
            node_quat = quaternion2rot(node_quat)
            gt_quat = quaternion2rot(gt_quat)
            edge_attr = quaternion2rot(edge_attr)

        return Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node)

class BaseDataset_MPLS(Dataset):
    def __init__(self, path=None, rot_type="qaut",on_disk=False) -> None:
        super().__init__()
        self.path = path
        self.rot_type = rot_type
        self.spt_gen = QuaternionSPT()
        self.on_disk = on_disk
        self.samples = self.collect_samples()

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self
    @abstractmethod
    def collect_samples(self):
        pass

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) :
        cache_path , i = self.samples[index]
        node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x_mpls" % (i+1), filename=cache_path), dtype=torch.float)
        gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)
        node_quat = node_quat.transpose(1,2)
        node_quat = rot2quaternion(node_quat)
        if self.rot_type == "rot":
            node_quat = quaternion2rot(node_quat)
            gt_quat = quaternion2rot(gt_quat)
            edge_attr = quaternion2rot(edge_attr)

        return Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node)


onedsfm_path = "/home/liheng03/projects/mra_opt/data/1dsfm"
syth_path = "/home/liheng03/projects/mra_opt/data/syth_dataset"
class Onedsfm(BaseDataset):
    def __init__(self, base_path=onedsfm_path, includes=None, excludes=None, 
            clean_cache=False, rot_type="quat", valid_edge="cc", spt_ratio=0.1,
            on_disk=False):
        self.scene_list = self.make_scene_list()
        if excludes is not None:
            for name in excludes:
                if name in self.scene_list:
                    self.scene_list.remove(name)
        if includes is not None:
            self.scene_list = includes
        self.valid_edge = valid_edge
        self.remove_cache = clean_cache
        self.rot_type=rot_type
        self.spt_ratio = spt_ratio
        super(Onedsfm, self).__init__(path=base_path, rot_type=rot_type, on_disk=on_disk)
    @classmethod
    def make_scene_list(cls):
        # exclude Trafalgar
        # and  ,
        scene_list = ["Alamo",
                            "Ellis_Island",
                            "Gendarmenmarkt",
                            "Madrid_Metropolis",
                            "Montreal_Notre_Dame",
                            "Notre_Dame",
                            "NYC_Library",
                            "Piazza_del_Popolo",
                            "Roman_Forum",
                            "Tower_of_London",
                            "Union_Square",
                            "Vienna_Cathedral",
                            "Yorkminster",
                            "Piccadilly",
                            "Trafalgar"]
        return scene_list

    def collect_samples(self):
        samples = []
        for scene_name in self.scene_list:
            if self.remove_cache:
                self.clean_cache(scene_name)

            if self.check_cache(scene_name):
                sample = self.load_cache(scene_name)
            else:
                sample = self.load_from_file(scene_name)
                self.save_cache(scene_name, sample)
            samples += sample
        return samples

    def load_from_file(self, scene_name):
        scene_path = os.path.join(self.path, scene_name)
        bundle_file = os.path.join(scene_path, "gt_bundle.out")
        cc_file = os.path.join(scene_path, "cc.txt")
        egs_file = os.path.join(scene_path, "EGs.txt")
        
        bundle = sfminit.Bundle.from_file(bundle_file)
        cc = sfminit.bundletypes.read_cc_file(cc_file)
        egs = sfminit.sfminittypes.read_EGs_file(egs_file)
        graph_data = graph_object(cc, egs,bundle,self.valid_edge).make_graph(self.rot_type)
        sample = []
        start_nodes_num = int(graph_data.gt_id.sum() * self.spt_ratio)
        gt_node_id = torch.arange(0, graph_data.x.shape[0], step=1)[graph_data.gt_id]
        start_nodes = gt_node_id[torch.randperm(graph_data.gt_id.sum())[:start_nodes_num]]
        with tqdm(total=start_nodes_num) as t:
            for start_node in start_nodes:
                t.set_description(desc="Collecting %s, Generating SPT" % scene_name)
                start_node, node_spt_quat = self.spt_gen(graph_data.edge_attr, graph_data.edge_index, graph_data.gt_rot, graph_data.gt_id, start_node.item())

                now_gt_rel_R = graph_data.gt_rot[start_node].clone().view(1,4).repeat(graph_data.x.shape[0],1)
                now_gt_rot = qmul(graph_data.gt_rot.clone(), inv_q(now_gt_rel_R))
                sample.append(Data(x=node_spt_quat.clone(), gt_rot=now_gt_rot.clone(), gt_id=graph_data.gt_id.clone(), edge_index=graph_data.edge_index.clone(), edge_attr=graph_data.edge_attr.clone(), valid_edge=graph_data.valid_edge.clone(), start_node=start_node))
                t.update()

        return sample

    def clean_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            os.remove(cache_name)

    def save_cache(self, scene, obj):
        # path = os.path.join(path,"cache.p")
        # with open(path, "wb") as fout:
        #     pickle.dump(obj, fout)
        cache_name = self.make_cache_name(scene)
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=graph_data.start_node)
        hf.close()

    def check_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            return True
        else :
            return False
    
    def load_cache(self, scene):
        # cache_path = self.make_cache_name(scene)

        # data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        # sample = []
        # with tqdm(total=data_len) as t:
        #     for i in range(data_len):
        #         t.set_description(desc="Collecting %s, Loading SPT" % scene)
        #         node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        #         edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        #         edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        #         valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        #         start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)
        #         sample.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
        #         t.update()
        cache_path = self.make_cache_name(scene)

        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s, Loading SPT" % scene)
                sample.append((cache_path, i))
                t.update()
        return sample

    def make_cache_name(self, scene):
        cache_path = os.path.join(self.path, scene, "cache_%s_%s_%.2f.h5" % (self.valid_edge, self.rot_type, self.spt_ratio))
        return cache_path

class Onedsfm_MPLS_Init(BaseDataset_MPLS):
    def __init__(self, base_path=onedsfm_path, includes=None, excludes=None, 
            clean_cache=False, rot_type="quat", valid_edge="cc", spt_ratio=0.1,
            on_disk=False):
        self.scene_list = self.make_scene_list()
        if excludes is not None:
            for name in excludes:
                if name in self.scene_list:
                    self.scene_list.remove(name)
        if includes is not None:
            self.scene_list = includes
        self.valid_edge = valid_edge
        self.remove_cache = clean_cache
        self.rot_type=rot_type
        self.spt_ratio = spt_ratio
        super(Onedsfm_MPLS_Init, self).__init__(path=base_path, rot_type=rot_type, on_disk=on_disk)
    @classmethod
    def make_scene_list(cls):
        # exclude Trafalgar
        # and  ,
        scene_list = ["Alamo",
                            "Ellis_Island",
                            "Gendarmenmarkt",
                            "Madrid_Metropolis",
                            "Montreal_Notre_Dame",
                            "Notre_Dame",
                            "NYC_Library",
                            "Piazza_del_Popolo",
                            "Roman_Forum",
                            "Tower_of_London",
                            "Union_Square",
                            "Vienna_Cathedral",
                            "Yorkminster"
                            # "Piccadilly"
                            # "Trafalgar"
                            ]
        return scene_list

    def collect_samples(self):
        samples = []
        for scene_name in self.scene_list:
            if self.remove_cache:
                self.clean_cache(scene_name)

            if self.check_cache(scene_name):
                sample = self.load_cache(scene_name)
            else:
                sample = self.load_from_file(scene_name)
                self.save_cache(scene_name, sample)
            samples += sample
        return samples

    def load_from_file(self, scene_name):
        scene_path = os.path.join(self.path, scene_name)
        bundle_file = os.path.join(scene_path, "gt_bundle.out")
        cc_file = os.path.join(scene_path, "cc.txt")
        egs_file = os.path.join(scene_path, "EGs.txt")
        
        bundle = sfminit.Bundle.from_file(bundle_file)
        cc = sfminit.bundletypes.read_cc_file(cc_file)
        egs = sfminit.sfminittypes.read_EGs_file(egs_file)
        graph_data = graph_object(cc, egs,bundle,self.valid_edge).make_graph(self.rot_type)
        sample = []
        # start_nodes_num = int(graph_data.gt_id.sum() * self.spt_ratio)
        # gt_node_id = torch.arange(0, graph_data.x.shape[0], step=1)[graph_data.gt_id]
        # start_nodes = gt_node_id[torch.randperm(graph_data.gt_id.sum())[:start_nodes_num]]
        # with tqdm(total=start_nodes_num) as t:
        #     for start_node in start_nodes:
        #         t.set_description(desc="Collecting %s, Generating SPT" % scene_name)
        #         start_node, node_spt_quat = self.spt_gen(graph_data.edge_attr, graph_data.edge_index, graph_data.gt_rot, graph_data.gt_id, start_node.item())

        #         now_gt_rel_R = graph_data.gt_rot[start_node].clone().view(1,4).repeat(graph_data.x.shape[0],1)
        #         now_gt_rot = qmul(graph_data.gt_rot.clone(), inv_q(now_gt_rel_R))
        #         sample.append(Data(x=node_spt_quat.clone(), gt_rot=now_gt_rot.clone(), gt_id=graph_data.gt_id.clone(), edge_index=graph_data.edge_index.clone(), edge_attr=graph_data.edge_attr.clone(), valid_edge=graph_data.valid_edge.clone(), start_node=start_node))
        #         t.update()

        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        valid_edge = graph_data.valid_edge

        # # sorted_index, sorted_id = torch.sort(edge_index, dim=0)
        # sorted_edge_index = sorted_index
        # sorted_edge_attr = edge_attr[sorted_id]        
        # sorted_valid_edge = valid_edge[sorted_id]
        N = graph_data.gt_rot.shape[0]
        adj_mat = torch.zeros(N,N).bool()
        adj_index = torch.zeros(N,N,2).long()
        adj_attr = torch.zeros(N,N,4)
        adj_valid = torch.zeros(N,N,1).bool()

        adj_mat[edge_index[0], edge_index[1]] = True
        adj_index[edge_index[0], edge_index[1]] = edge_index.T
        adj_attr[edge_index[0], edge_index[1]] = edge_attr
        adj_valid[edge_index[0], edge_index[1]] = valid_edge

        sorted_edge_index = []
        sorted_edge_attr = []
        sorted_valid_edge = []
        for i in range(N):
            for j in range(i,N):
                if adj_mat[i,j].item() is True:
                    sorted_edge_index.append(adj_index[i,j])
                    sorted_edge_attr.append(adj_attr[i,j])
                    sorted_valid_edge.append(adj_valid[i,j])
        sorted_edge_index = torch.stack(sorted_edge_index, dim=0).T
        sorted_edge_attr = torch.stack(sorted_edge_attr, dim=0)
        sorted_valid_edge = torch.stack(sorted_valid_edge, dim=0)

        sample.append(Data(x=torch.zeros_like(graph_data.gt_rot),
                            gt_rot=graph_data.gt_rot,
                            gt_id=graph_data.gt_id,
                            edge_index=sorted_edge_index,
                            edge_attr=sorted_edge_attr,
                            valid_edge=sorted_valid_edge,
                            start_node=0,
                            ))

        return sample

    def clean_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            os.remove(cache_name)

    def save_cache(self, scene, obj):
        # path = os.path.join(path,"cache.p")
        # with open(path, "wb") as fout:
        #     pickle.dump(obj, fout)
        cache_name = self.make_cache_name(scene)
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            # hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/mpls/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            # hf.create_dataset("/mpls/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/mpls/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/mpls/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            # hf.create_dataset("/mpls/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            # hf.create_dataset("/mpls/%d/start_node" % (i+1), data=graph_data.start_node)
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())

            edge_index = torch.cat([graph_data.edge_index, torch.stack([graph_data.edge_index[1], graph_data.edge_index[0]], dim=0)], dim=1)
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=edge_index.data.cpu().numpy())
            edge_attr = torch.cat([graph_data.edge_attr, inv_q(graph_data.edge_attr)], dim=0)
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=edge_attr.cpu().numpy())
            valid_edge = torch.cat([graph_data.valid_edge,graph_data.valid_edge], dim=0)
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=graph_data.start_node)
        hf.close()

    def call_matlab_mpls_for_init(filename):
        pass

    def check_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            return True
        else :
            return False
    
    def load_cache(self, scene):
        # cache_path = self.make_cache_name(scene)

        # data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        # sample = []
        # with tqdm(total=data_len) as t:
        #     for i in range(data_len):
        #         t.set_description(desc="Collecting %s, Loading SPT" % scene)
        #         node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        #         edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        #         edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        #         valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        #         start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)
        #         sample.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
        #         t.update()
        cache_path = self.make_cache_name(scene)

        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s, Loading SPT" % scene)
                sample.append((cache_path, i))
                t.update()
        return sample

    def make_cache_name(self, scene):
        cache_path = os.path.join(self.path, scene, "cache_%s_%s_%.2f_mpls_init.h5" % (self.valid_edge, self.rot_type, self.spt_ratio))
        return cache_path

class OnedsfmNoInit(BaseDataset):
    def __init__(self, base_path=onedsfm_path, includes=None, excludes=None, 
            clean_cache=False, rot_type="quat", valid_edge="cc", spt_ratio=0.1,
            on_disk=False):
        self.scene_list = self.make_scene_list()
        if excludes is not None:
            for name in excludes:
                if name in self.scene_list:
                    self.scene_list.remove(name)
        if includes is not None:
            self.scene_list = includes
        self.valid_edge = valid_edge
        self.remove_cache = clean_cache
        self.rot_type=rot_type
        self.spt_ratio = spt_ratio
        super(OnedsfmNoInit, self).__init__(path=base_path, rot_type=rot_type, on_disk=on_disk)
    @classmethod
    def make_scene_list(cls):
        # exclude Trafalgar
        # and   "Piccadilly",
        scene_list = ["Alamo",
                            "Ellis_Island",
                            "Gendarmenmarkt",
                            "Madrid_Metropolis",
                            "Montreal_Notre_Dame",
                            "Notre_Dame",
                            "NYC_Library",
                            "Piazza_del_Popolo",
                            "Roman_Forum",
                            "Tower_of_London",
                            "Union_Square",
                            "Vienna_Cathedral",
                            "Yorkminster",
                            "Piccadilly",
                            "Trafalgar"]
        return scene_list

    def collect_samples(self):
        samples = []
        for scene_name in self.scene_list:
            if self.remove_cache:
                self.clean_cache(scene_name)

            if self.check_cache(scene_name):
                sample = self.load_cache(scene_name)
            else:
                sample = self.load_from_file(scene_name)
                self.save_cache(scene_name, sample)
            samples += sample
        return samples

    def load_from_file(self, scene_name):
        scene_path = os.path.join(self.path, scene_name)
        bundle_file = os.path.join(scene_path, "gt_bundle.out")
        cc_file = os.path.join(scene_path, "cc.txt")
        egs_file = os.path.join(scene_path, "EGs.txt")
        
        bundle = sfminit.Bundle.from_file(bundle_file)
        cc = sfminit.bundletypes.read_cc_file(cc_file)
        egs = sfminit.sfminittypes.read_EGs_file(egs_file)
        print("Loading %s"% scene_name)
        graph_data = graph_object(cc, egs,bundle,self.valid_edge).make_graph(self.rot_type)
        sample = []
        graph_data.start_node=0
        sample.append(graph_data)

        return sample

    def clean_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            os.remove(cache_name)

    def save_cache(self, scene, obj):
        # path = os.path.join(path,"cache.p")
        # with open(path, "wb") as fout:
        #     pickle.dump(obj, fout)
        cache_name = self.make_cache_name(scene)
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=graph_data.start_node)
        hf.close()

    def check_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            return True
        else :
            return False
    
    def load_cache(self, scene):
        # cache_path = self.make_cache_name(scene)

        # data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        # sample = []
        # with tqdm(total=data_len) as t:
        #     for i in range(data_len):
        #         t.set_description(desc="Collecting %s, Loading SPT" % scene)
        #         node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        #         edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        #         edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        #         valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        #         start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)
        #         sample.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
        #         t.update()
        cache_path = self.make_cache_name(scene)

        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s, Loading SPT" % scene)
                sample.append((cache_path, i))
                t.update()
        return sample

    def make_cache_name(self, scene):
        cache_path = os.path.join(self.path, scene, "cache_%s_%s_%.2f_no_init.h5" % (self.valid_edge, self.rot_type, self.spt_ratio))
        return cache_path

class Yfcc100(BaseDataset):
    def __init__(self, path,
                rot_type="quat",
                remove_cache=False,
                on_disk=False,
                lens = 53) -> None:
        self.remove_cache = remove_cache
        self.path = path
        self.lens = lens
        super(Yfcc100, self).__init__(path=path,rot_type=rot_type,on_disk=on_disk)
    
    def collect_samples(self):
        samples = []
        if self.remove_cache:
            self.clean_cache()
        if self.check_cache():
            samples = self.load_cache()
        else:
            samples = self.load_from_file()
            self.save_cache(samples)
        return samples

    def make_cache_name(self):
        file_name = os.path.split(self.path)[-1]
        return os.path.join(os.path.split(self.path)[0], "mra_h5_data_" + file_name) 

    def clean_cache(self):
        cache_name = self.make_cache_name()
        if os.path.isfile(cache_name):
            os.remove(cache_name)
    def check_cache(self):
        cache_name = self.make_cache_name()
        if os.path.isfile(cache_name):
            return True
        else :
            return False

    def load_cache(self):
        cache_path = self.make_cache_name()
        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s, Loading SPT" % (self.path))
                sample.append((cache_path, i))
                t.update()
        return sample

    def load_from_file(self):
        filename = self.path
        samples = []
        for item in range(self.lens):
            x = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/x', filename=filename, options=None), dtype=torch.float)
            xt = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/xt', filename=filename, options=None), dtype=torch.float)
            o = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/o', filename=filename, options=None), dtype=torch.float)
            o =o.view(1,o.shape[0])
        #   onode = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/onode', filename=filename, options=None), dtype=torch.float)
        #   omarker = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/omarker', filename=filename, options=None), dtype=torch.float)
            edge_index = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_index', filename=filename, options=None), dtype=torch.long).T
            edge_attr = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_feature', filename=filename, options=None), dtype=torch.float)
            edge_attr = inv_q(edge_attr)

            edge_index_g = edge_index.clone()
            edge_attr_g = edge_attr.clone()
            edge_index_g[:,::2] = edge_index[:, :edge_index.shape[1]//2]
            edge_index_g[:,1::2] = edge_index[:, edge_index.shape[1]//2:]

            edge_attr_g[::2,:] = edge_attr[0:edge_attr.shape[0]//2,:]
            edge_attr_g[1::2,:] = edge_attr[edge_attr.shape[0]//2:, :]

            edge_index = edge_index_g
            edge_attr = edge_attr_g
            y = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/y', filename=filename, options=None), dtype=torch.float)
            samples.append(Data(x=x.clone(),gt_rot=y.clone(), gt_id=torch.ones(y.shape[0]).long() ,edge_index=edge_index, edge_attr=edge_attr, valid_edge=torch.ones(edge_attr.shape[0]).long()))
        return samples

    def save_cache(self, obj):
        cache_name = self.make_cache_name()
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=0)
        hf.close()

class SythDataset(BaseDataset):
    def __init__(self, base_path=syth_path, 
                prefix="test_data", 
                dataset_split_num=12, 
                rot_type="quat", 
                remove_cache=False,
                get_part="train",
                spt_ratio=0.1,
                on_disk=False):
        self.rot_type=rot_type
        self.remove_cache = remove_cache
        self.spt_ratio = spt_ratio
        self.prefix = prefix
        self.dataset_split_num = dataset_split_num
        self.get_part = get_part
        self.train_list, self.eval_list, self.test_list = self.make_scene_list()
        super(SythDataset, self).__init__(path=base_path, rot_type=rot_type,on_disk=on_disk)

    def make_scene_list(self):
        eval_list = [self.prefix + ("_%d.h5" % 1)]
        test_list = [self.prefix + ("_%d.h5" % 2)]
        train_list = [self.prefix + ("_%d.h5" % i) for i in range(3, self.dataset_split_num+1)]
        return train_list, eval_list, test_list

    def collect_samples(self):
        train_samples = []

        now_list = None
        if self.get_part == "train":
            now_list = self.train_list
        elif self.get_part == "eval":
            now_list = self.eval_list
        elif self.get_part == "test":
            now_list = self.test_list

        if self.remove_cache:
            self.clean_cache(self.get_part)
        if self.check_cache(self.get_part):
            samples = self.load_cache(self.get_part)
        else:
            samples = self.load_from_file(now_list)
            self.save_cache(self.get_part, samples)
        return samples

    def load_from_file(self, scene_list):
        samples = []
        for scene_name in scene_list:
            scene_path = os.path.join(self.path, scene_name)
            data_len = int(hdf5storage.read(path="/data/lens", filename=scene_path).item())
            with tqdm(total=data_len) as t:
                subt = tqdm(total=0)
                for i in range(data_len):
                    t.set_description(desc="Collecting %s, Generating SPT" % scene_name)
                    node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=scene_path), dtype=torch.float)
                    gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/y" % (i+1), filename=scene_path), dtype=torch.float)
                    edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=scene_path), dtype=torch.long).T
                    edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_feature" % (i+1), filename=scene_path), dtype=torch.float)
                    gt_id = torch.ones(node_quat.shape[0]).bool()
                    valid_edge = torch.ones(edge_index.shape[1]).bool().view(-1,1)
                    cc = [i for i in range(node_quat.shape[0])]
                    start_node_num = int(node_quat.shape[0] * self.spt_ratio)
                    start_node_num = max(3, start_node_num)
                    gt_node_id = torch.arange(0, node_quat.shape[0], step=1)[gt_id]
                    start_nodes = gt_node_id[torch.randperm(node_quat.shape[0])][:start_node_num]
                    subt.total = start_node_num
                    for start_node in start_nodes:
                        start_node, node_spt_quat = self.spt_gen(edge_attr, edge_index, gt_quat, gt_id, start_node.item())
                        now_gt_rel_R = gt_quat[start_node].clone().view(1,4).repeat(node_quat.shape[0],1)
                        now_gt_rot = qmul(gt_quat.clone(), inv_q(now_gt_rel_R))
                        samples.append(Data(x=node_spt_quat.clone(), gt_rot=now_gt_rot.clone(), gt_id=gt_id.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone(), valid_edge=valid_edge.clone(), start_node=start_node))
                        subt.update()
                    subt.reset()
                    # samples.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
                    t.update()
        return samples

    def clean_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            os.remove(cache_name)

    def save_cache(self, scene, obj):
        cache_name = self.make_cache_name(scene)
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=graph_data.start_node)
        hf.close()

    def check_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            return True
        else :
            return False
    
    def load_cache(self, scene):
        # cache_path = self.make_cache_name(scene)

        # data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        # sample = []
        # with tqdm(total=data_len) as t:
        #     for i in range(data_len):
        #         t.set_description(desc="Collecting %s_%s, Loading SPT" % (self.prefix, scene))
        #         node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/gt_rot" % (i+1), filename=cache_path), dtype=torch.float)
        #         gt_id = torch.tensor(hdf5storage.read(path="/data/%d/gt_id" % (i+1), filename=cache_path), dtype=torch.bool)
        #         edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=cache_path), dtype=torch.long)
        #         edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_attr" % (i+1), filename=cache_path), dtype=torch.float)
        #         valid_edge = torch.tensor(hdf5storage.read(path="/data/%d/valid_edge" % (i+1), filename=cache_path), dtype=torch.bool)
        #         start_node = torch.tensor(hdf5storage.read(path="/data/%d/start_node" % (i+1), filename=cache_path), dtype=torch.long)
        #         sample.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
        #         t.update()
        # return sample
        
        cache_path = self.make_cache_name(scene)

        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s_%s, Loading SPT" % (self.prefix, scene))
                sample.append((cache_path, i))
                t.update()
        if scene == "eval":
            sample = sample[:20]
        return sample

    def make_cache_name(self, data_split="train"):
        return os.path.join(self.path, self.prefix + "_%s.h5" % data_split)
    

class SythDatasetNoInit(BaseDataset):
    def __init__(self, base_path=syth_path, 
                prefix="test_data", 
                dataset_split_num=12, 
                rot_type="quat", 
                remove_cache=False,
                get_part="train",
                spt_ratio=0.1,
                on_disk=False):
        self.rot_type=rot_type
        self.remove_cache = remove_cache
        self.spt_ratio = spt_ratio
        self.prefix = prefix
        self.dataset_split_num = dataset_split_num
        self.get_part = get_part
        self.train_list, self.eval_list, self.test_list = self.make_scene_list()
        super(SythDatasetNoInit, self).__init__(path=base_path, rot_type=rot_type,on_disk=on_disk)

    def make_scene_list(self):
        eval_list = [self.prefix + ("_%d.h5" % 1)]
        test_list = [self.prefix + ("_%d.h5" % 2)]
        train_list = [self.prefix + ("_%d.h5" % i) for i in range(3, self.dataset_split_num+1)]
        return train_list, eval_list, test_list

    def collect_samples(self):
        train_samples = []

        now_list = None
        if self.get_part == "train":
            now_list = self.train_list
        elif self.get_part == "eval":
            now_list = self.eval_list
        elif self.get_part == "test":
            now_list = self.test_list

        if self.remove_cache:
            self.clean_cache(self.get_part)
        if self.check_cache(self.get_part):
            samples = self.load_cache(self.get_part)
        else:
            samples = self.load_from_file(now_list)
            self.save_cache(self.get_part, samples)
        return samples

    def load_from_file(self, scene_list):
        samples = []
        for scene_name in scene_list:
            scene_path = os.path.join(self.path, scene_name)
            data_len = int(hdf5storage.read(path="/data_lens", filename=scene_path).item())
            with tqdm(total=data_len) as t:
                for i in range(data_len):
                    t.set_description(desc="Collecting %s, Generating SPT" % scene_name)
                    node_quat = torch.tensor(hdf5storage.read(path="/data/%d/x" % (i+1), filename=scene_path), dtype=torch.float)
                    gt_quat = torch.tensor(hdf5storage.read(path="/data/%d/y" % (i+1), filename=scene_path), dtype=torch.float)
                    edge_index = torch.tensor(hdf5storage.read(path="/data/%d/edge_index" % (i+1), filename=scene_path), dtype=torch.long).T
                    edge_attr = torch.tensor(hdf5storage.read(path="/data/%d/edge_feature" % (i+1), filename=scene_path), dtype=torch.float)
                    gt_id = torch.ones(node_quat.shape[0]).bool()
                    valid_edge = torch.ones(edge_index.shape[1]).bool().view(-1,1)
                    cc = [i for i in range(node_quat.shape[0])]
                    start_node_num = int(node_quat.shape[0] * self.spt_ratio)
                    start_node_num = max(3, start_node_num)
                    gt_node_id = torch.arange(0, node_quat.shape[0], step=1)[gt_id]
                    start_nodes = gt_node_id[torch.randperm(node_quat.shape[0])][:start_node_num]
                    samples.append(Data(x=node_quat.clone(), gt_rot=gt_quat.clone(), gt_id=gt_id.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone(), valid_edge=valid_edge.clone(), start_node=0))
                    # samples.append(Data(x=node_quat, gt_rot=gt_quat, gt_id=gt_id, edge_index=edge_index, edge_attr=edge_attr, valid_edge=valid_edge, start_node=start_node))
                    t.update()
        return samples

    def clean_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            os.remove(cache_name)

    def save_cache(self, scene, obj):
        cache_name = self.make_cache_name(scene)
        hf = h5py.File(cache_name, "w")

        data_len = len(obj)
        hf.create_dataset("/data_len", data=data_len)

        for i in range(data_len):
            graph_data = obj[i]
            hf.create_dataset("/data/%d/x" % (i+1), data=graph_data.x.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_rot" % (i+1), data=graph_data.gt_rot.data.cpu().numpy())
            hf.create_dataset("/data/%d/gt_id" % (i+1), data=graph_data.gt_id.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_index" % (i+1), data=graph_data.edge_index.data.cpu().numpy())
            hf.create_dataset("/data/%d/edge_attr" % (i+1), data=graph_data.edge_attr.cpu().numpy())
            hf.create_dataset("/data/%d/valid_edge" % (i+1), data=graph_data.valid_edge.data.cpu().numpy())
            hf.create_dataset("/data/%d/start_node" % (i+1), data=graph_data.start_node)
        hf.close()

    def check_cache(self, scene):
        cache_name = self.make_cache_name(scene)
        if os.path.isfile(cache_name):
            return True
        else :
            return False
    
    def load_cache(self, scene):        
        cache_path = self.make_cache_name(scene)

        data_len = hdf5storage.read(path="/data_len", filename=cache_path)
        sample = []
        with tqdm(total=data_len) as t:
            for i in range(data_len):
                t.set_description(desc="Collecting %s_%s, Loading SPT" % (self.prefix, scene))
                sample.append((cache_path, i))
                t.update()
        return sample

    def make_cache_name(self, data_split="train"):
        return os.path.join(self.path, self.prefix + "_%s_no_init.h5" % data_split)
    


# 
# O = 1dsfm
# Y = yfcc100
# S = syth
# onedsfm_path = "./data/ondsfm"

def fetch_dataset(args):
    train_ds = args.dataset_setting.train_ds
    train_datasets = None
    if "O" in train_ds:
        train_dataset = Onedsfm(args.dataset_setting.onedsfm.base_path, 
                                includes=args.dataset_setting.onedsfm.includes,
                                excludes=args.dataset_setting.onedsfm.excludes,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        # train_datasets[0].make_graph()
        if train_datasets is None:
            train_datasets = 10 * train_dataset
        else: 
            train_datasets = train_datasets + 10 * train_dataset
    if "O_MPLS" in train_ds:
        train_dataset = Onedsfm_MPLS_Init(args.dataset_setting.onedsfm.base_path, 
                                includes=args.dataset_setting.onedsfm.includes,
                                excludes=args.dataset_setting.onedsfm.excludes,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        if train_datasets is None:
            train_datasets = train_dataset
        else:
            train_datasets = train_datasets + train_dataset
    if "ONI" in train_ds:
        train_dataset = OnedsfmNoInit(args.dataset_setting.onedsfm.base_path, 
                                includes=args.dataset_setting.onedsfm.includes,
                                excludes=args.dataset_setting.onedsfm.excludes,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        # train_datasets[0].make_graph()
        if train_datasets is None:
            train_datasets = train_dataset
        else: 
            train_datasets = train_datasets +  train_dataset
    if "Y" in train_ds:
        pass
    if "S" in train_ds:
        train_dataset = SythDataset(base_path=args.dataset_setting.syth.base_path,
                                    prefix=args.dataset_setting.syth.prefix,
                                    dataset_split_num=args.dataset_setting.syth.dataset_split_num,
                                    rot_type=args.dataset_setting.syth.rot_type,
                                    remove_cache=args.dataset_setting.syth.clean_cache,
                                    get_part=args.dataset_setting.syth.get_part,
                                    spt_ratio=args.dataset_setting.syth.spt_ratio)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 
    if "SNI" in train_ds:
        for key in args.dataset_setting.keys():
            if "syth" in key:
                train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part,
                                            spt_ratio=args.dataset_setting[key].spt_ratio)
                if train_datasets is None:
                    train_datasets =  train_dataset 
                else:
                    train_datasets = train_datasets  +  train_dataset       
    if "ABL_0" in train_ds:
        key = "ABL_0"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_1" in train_ds:
        key = "ABL_1"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_2" in train_ds:
        key = "ABL_2"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_3" in train_ds:
        key = "ABL_3"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_4" in train_ds:
        key = "ABL_4"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_5" in train_ds:
        key = "ABL_5"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_6" in train_ds:
        key = "ABL_6"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_7" in train_ds:
        key = "ABL_7"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "ABL_8" in train_ds:
        key = "ABL_8"
        train_dataset = SythDatasetNoInit(base_path=args.dataset_setting[key].base_path,
                                            prefix=args.dataset_setting[key].prefix,
                                            dataset_split_num=args.dataset_setting[key].dataset_split_num,
                                            rot_type=args.dataset_setting[key].rot_type,
                                            remove_cache=args.dataset_setting[key].clean_cache,
                                            get_part=args.dataset_setting[key].get_part)
        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 

    if "YFCC_2" in train_ds:
        key = "YFCC_2"
        train_dataset = Yfcc100(path="/data/NeuRoRA_overfit/data/train_all_yfcc_2.h5",
                                rot_type="quat",
                                remove_cache=False,
                                on_disk=False,
                                lens=53)

        if train_datasets is None:
            train_datasets =  train_dataset 
        else:
            train_datasets = train_datasets  +  train_dataset 


    train_loader = DataLoader(train_datasets, batch_size=args.hypers_params.train_batch_size,
                            pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    return train_loader

def fetch_eval_dataset(args):
    eval_datasets = None
    if "O" in args.eval:
        onedsfm_datasets = Onedsfm(args.dataset_setting.onedsfm.base_path, 
                                includes=args.eval.O,
                                excludes=None,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        eval_datasets = onedsfm_datasets
    if "O_MPLS" in args.eval:
        onedsfm_datasets = Onedsfm_MPLS_Init(args.dataset_setting.onedsfm.base_path, 
                                includes=args.eval.O_MPLS,
                                excludes=None,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        eval_datasets = onedsfm_datasets
    if "ONI" in args.eval:
        onedsfm_datasets = OnedsfmNoInit(args.dataset_setting.onedsfm.base_path, 
                                includes=args.eval.ONI,
                                excludes=None,
                                clean_cache=args.dataset_setting.onedsfm.clean_cache,
                                rot_type=args.dataset_setting.onedsfm.rot_type,
                                valid_edge=args.dataset_setting.onedsfm.valid_edge,
                                spt_ratio=args.dataset_setting.onedsfm.spt_ratio)
        eval_datasets = onedsfm_datasets
    if "S" in args.eval:
        syth_dataset = SythDataset(base_path=args.dataset_setting.syth.base_path,
                                    prefix=args.dataset_setting.syth.prefix,
                                    dataset_split_num=args.dataset_setting.syth.dataset_split_num,
                                    rot_type=args.dataset_setting.syth.rot_type,
                                    remove_cache=args.dataset_setting.syth.clean_cache,
                                    get_part="eval",
                                    spt_ratio=args.dataset_setting.syth.spt_ratio)
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "SNI" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.SNI].base_path,
                                    prefix=args.dataset_setting[args.eval.SNI].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.SNI].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.SNI].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.SNI].clean_cache,
                                    get_part="eval",
                                    spt_ratio=args.dataset_setting[args.eval.SNI].spt_ratio)
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    
    if "ABL_0" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_0].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_0].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_0].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_0].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_0].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_2" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_2].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_2].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_2].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_2].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_2].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_1" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_1].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_1].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_1].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_1].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_1].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_3" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_3].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_3].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_3].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_3].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_3].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_4" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_4].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_4].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_4].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_4].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_4].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_5" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_5].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_5].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_5].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_5].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_5].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset

    if "ABL_6" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_6].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_6].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_6].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_6].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_6].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_7" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_7].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_7].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_7].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_7].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_7].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_7" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_7].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_7].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_7].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_7].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_7].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "ABL_8" in args.eval:
        syth_dataset = SythDatasetNoInit(base_path=args.dataset_setting[args.eval.ABL_8].base_path,
                                    prefix=args.dataset_setting[args.eval.ABL_8].prefix,
                                    dataset_split_num=args.dataset_setting[args.eval.ABL_8].dataset_split_num,
                                    rot_type=args.dataset_setting[args.eval.ABL_8].rot_type,
                                    remove_cache=args.dataset_setting[args.eval.ABL_8].clean_cache,
                                    get_part="eval")
        if eval_datasets is None:
            eval_datasets = syth_dataset
        else:
            eval_datasets = eval_datasets + syth_dataset
    if "YFCC_2" in args.eval:
        key = "YFCC_2"
        yfcc_datasets = Yfcc100(path="/data/NeuRoRA_overfit/data/valid_all_yfcc_2.h5",
                                rot_type="quat",
                                remove_cache=False,
                                on_disk=False,
                                lens=14)
        if eval_datasets is None:
            eval_datasets = yfcc_datasets
        else:
            eval_datasets = eval_datasets + yfcc_datasets
    eval_loader = DataLoader(eval_datasets,batch_size=args.hypers_params.eval_batch_size,
                            pin_memory=False, shuffle=False, num_workers=4, drop_last=False)
    return eval_loader

