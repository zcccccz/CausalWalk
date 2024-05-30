import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import json
import logging
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
eps = 1e-12

def init_logger(log_file=None):
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

# 合并evidence图和claim图, claim中每个结点与evidence中每个结点相连接。
def merage_graph(claim_graph, evidence_graph):
    x1 = claim_graph["nodes"]
    x2 = evidence_graph["nodes"]
    edges1 = claim_graph["edges"]
    edges2 = evidence_graph["edges"]
    edges1 = torch.cat([edges1,torch.ones((len(x1),len(x2)),device=edges1.device)],dim=1)
    edges2 = torch.cat([torch.ones((len(x2),len(x1)),device=edges2.device),edges2],dim=1)
    edges3 = torch.cat([edges1,edges2],dim=0)
    x3 = torch.cat([x1,x2],dim=0)
    graph = {
        "nodes":x3,
        "edges":edges3,
        "claim_length":len(x1)
    }
    return graph

def load_graph_data(file):
    a = torch.load(file)
    res = []
    for id in tqdm(list(a.keys())):
        label = a[id]["label"]
        graph = merage_graph(a[id]["claim_graph"], a[id]["evidences_graph"])
        graph["label"] = torch.LongTensor([label])
        graph["id"] = id
        res.append(graph)
    return res

def load_pair_data(file):
    a = torch.load(file)
    claims, evidences, claim_evidences, labels, evi_labels = a.values()
    return claims, claim_evidences, labels

def load_data(file): # concat sentence
    a = torch.load(file)
    indexs, labels, embeddings = a.values()
    return embeddings, labels

# 合并evidence图和claim图, claim中每个结点与evidence中每个结点相连接，合并所有的claim结点，只有一个claim结点。
def merage_graph_one_claim_node(claim_graph, evidence_graph):
    x1 = claim_graph["nodes"]
    x2 = evidence_graph["nodes"]
    x1 = x1.mean(dim=0).unsqueeze(0) # [1,768]
    edges1 = torch.ones((1,1))
    edges2 = evidence_graph["edges"]
    edges1 = torch.cat([edges1,torch.ones((len(x1),len(x2)),device=edges1.device)],dim=1)
    edges2 = torch.cat([torch.ones((len(x2),len(x1)),device=edges2.device),edges2],dim=1)
    edges3 = torch.cat([edges1,edges2],dim=0)
    x3 = torch.cat([x1,x2],dim=0)
    graph = {
        "nodes":x3,
        "edges":edges3,
        "claim_length":len(x1)
    }
    return graph

def load_graph_data_one_claim_node(file):
    a = torch.load(file)
    res = []
    for id in tqdm(list(a.keys())):
        label = a[id]["label"]
        graph = merage_graph_one_claim_node(a[id]["claim_graph"], a[id]["evidences_graph"])
        graph["label"] = torch.LongTensor([label])
        res.append(graph)
    return res

def cal_kmeans_center(graphs, n_cluster):
    all_graph_rep = []
    for graph in graphs:
        x = graph["nodes"] # 结点特征 [200,768] 假如有200个结点
        claim_rep = x[0:graph["claim_length"]].mean(dim=0) # [768]
        evidence_rep = x[graph["claim_length"]:].mean(dim=0) # [768]
        graph_rep = torch.cat([claim_rep,evidence_rep],dim=0) # [768*2]
        all_graph_rep.append(graph_rep.unsqueeze(0))
    all_graph_rep = torch.cat(all_graph_rep,dim=0)
    all_graph_rep = all_graph_rep.numpy()

    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(all_graph_rep)
    centers = kmeans.cluster_centers_
    centers = torch.tensor(centers)
    labels = kmeans.labels_
    logits = []
    for i in range(n_cluster):
        logits.append(sum(labels==i))
    logits = torch.tensor(logits) / sum(logits)
    return centers, logits

def cal_nwgm_center(graphs, n_class):
    all_graph_rep = []
    for graph in graphs:
        x = graph["nodes"] # 结点特征 [200,768] 假如有200个结点
        claim_rep = x[0:graph["claim_length"]].mean(dim=0) # [768]
        evidence_rep = x[graph["claim_length"]:].mean(dim=0) # [768]
        graph_rep = torch.cat([claim_rep,evidence_rep],dim=0) # [768*2]
        all_graph_rep.append(graph_rep.unsqueeze(0))
    all_graph_rep = torch.cat(all_graph_rep,dim=0) # [100,768*2]

    z_list = []
    for i in range(n_class):
        z_i = []
        for j, graph in enumerate(graphs):
            label = graph["label"]
            if label == i:
                z_i.append(all_graph_rep[j].unsqueeze(0))
        z_i = torch.cat(z_i,dim=0)
        z_i = z_i.mean(dim=0)
        z_list.append(z_i.unsqueeze(0))
    z_list = torch.cat(z_list,dim=0) # [3,768*2]
    return z_list

from models import ONE_ATTENTION_with_bert
def cal_nwgm_center(model_path, dataloader, nclass, device, feature_num, evi_num):
    model = ONE_ATTENTION_with_bert(feature_num, nclass, evi_num)
    model = model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    graph_rep_list = []
    all_labels = []
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            data = [i.to(device) for i in data]
            input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
            graph_rep = model.cal_graph_representation(data) # [batch,768]
            graph_rep_list.append(graph_rep.detach())
            all_labels.append(labels.detach())
            description = "calculate graph representation..."
            dataloader.set_description(description)
    graph_rep_list = torch.cat(graph_rep_list,dim=0) # [sample_num,768]
    all_labels = torch.cat(all_labels,dim=0) # [sample_num

    z_list = []
    for i in range(nclass):
        z_i = []
        for j, graph in enumerate(graph_rep_list):
            label = all_labels[j]
            if label == i:
                z_i.append(graph_rep_list[j].unsqueeze(0))
        z_i = torch.cat(z_i,dim=0)
        z_i = z_i.mean(dim=0) # [768]
        z_list.append(z_i.unsqueeze(0))
    z_list = torch.cat(z_list,dim=0) # [3,768]
    return z_list

def cal_nwgm_center_attention(model_path, dataloader, nclass, device, feature_num, evi_num, n_cluster, politihop_2way=False):
    if politihop_2way: # now nclass == 2 but the graph rep model need nclass == 3
        model = ONE_ATTENTION_with_bert(feature_num, 3, evi_num)
    else:
        model = ONE_ATTENTION_with_bert(feature_num, nclass, evi_num)
    model = model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    graph_rep_list = []
    all_labels = []
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            data = [i.to(device) for i in data]
            input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
            graph_rep = model.cal_graph_representation(data) # [batch,768]
            graph_rep_list.append(graph_rep.detach())
            all_labels.append(labels.detach())
            description = "calculate graph representation..."
            dataloader.set_description(description)
    graph_rep_list = torch.cat(graph_rep_list,dim=0) # [sample_num,768]
    all_labels = torch.cat(all_labels,dim=0) # [sample_num]

    z_list = []
    for i in range(nclass):
        z_i = []
        for j, graph in enumerate(graph_rep_list):
            label = all_labels[j]
            if label == i:
                z_i.append(graph_rep_list[j].unsqueeze(0))
        z_i = torch.cat(z_i,dim=0) # [class_i_num,768]
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(z_i.cpu().numpy())
        labels = kmeans.labels_
        # centers = []
        # for center in range(n_cluster):
        #     center_mean = []
        #     for k in range(len(z_i)):
        #         if labels[k] == center:
        #             center_mean.append(z_i[k].unsqueeze(0))
        #     center_mean = torch.cat(center_mean,dim=0) # [cluster_size,768]
        #     center_mean = center_mean.mean(dim=0) # [768]
        #     centers.append(center_mean.unsqueeze(0))
        # centers = torch.cat(centers,dim=0) # [n_cluster, 768]

        centers = kmeans.cluster_centers_
        centers = torch.tensor(centers) # [n_cluster,768]
        # logits = []
        # for i in range(n_cluster):
        #     logits.append(sum(labels==i))
        # logits = torch.tensor(logits) / sum(logits)

        z_list.append(centers.unsqueeze(0))
    z_list = torch.cat(z_list,dim=0) # [3,n_cluster,768]
    return z_list.to(device)