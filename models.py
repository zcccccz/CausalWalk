import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, ELU, LeakyReLU, Sigmoid
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import heapq
from tqdm import tqdm
from sklearn.cluster import KMeans
from pytorch_pretrained_bert.modeling import BertModel
eps = 1e-12

class SelfAttention(nn.Module):
    def __init__(self, nhid):
        super(SelfAttention, self).__init__()
        self.nhid = nhid
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ELU(),
            Linear(64, 1),
            ELU(),
        )
    def forward(self, evidences, claims, evi_labels=None):  
        # evidences [256,5,768] claims [256,768] evi_labels [256,5]
        # claims = claims.unsqueeze(1).repeat(1,evidences.shape[1],1)  # [256,5,768]
        claims = claims.unsqueeze(1).expand(claims.shape[0],evidences.shape[1],claims.shape[-1])  # [256,5,768]
        temp = torch.cat((claims,evidences),dim=-1)  # [256,5,768*2]
        weight = self.project(temp)  # [256,5,1]
        if evi_labels is not None:
            # evi_labels = evi_labels[:,1:] # [batch,5]
            mask = evi_labels == 0 # [batch,5]
            mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) # 邻接矩阵中为0的地方填上负无穷 [batch,5]
            weight = weight + mask.unsqueeze(-1) # [256,5,1]
        weight = F.softmax(weight,dim=1)  # [256,5,1]
        outputs = torch.matmul(weight.transpose(1,2), evidences).squeeze(dim=1)  # [256,768]
        return outputs

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        self.alpha = self.alpha.to(target.device)
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class ONE_ATTENTION_with_bert(torch.nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num) -> None:
        super(ONE_ATTENTION_with_bert, self).__init__()
        self.evi_max_num = evi_max_num
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.attention = SelfAttention(nfeat*2)
        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat), # +1解释：第一个结点，图表示
            ELU(True),
            Linear(nfeat, nclass),
            ELU(True),
        )

    def cal_graph_representation(self, data):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.evi_max_num,pooled_output.shape[-1]) # [batch,6,768]
        datas = []
        for i in range(len(pooled_output)):
            x = pooled_output[i] # [6,768]
            # 全连接
            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        x, edge_index = datas.x, datas.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)
        x = x.view(-1,1+self.evi_max_num,x.shape[-1]) # [batch,6,768]
        feature_batch, claim_batch = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(feature_batch, claim_batch, sent_labels[:,1:]) # [batch,768]
        return graph_rep

    def forward(self, data):
        graph_rep = self.cal_graph_representation(data)
        outputs = self.classifier(graph_rep)
        return outputs

class Walk_with_bert(nn.Module):
    def __init__(self, nfeat, nclass, max_length, beam_size, max_evi_num, causal_method=None):
        super(Walk_with_bert, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.max_length = max_length
        self.beam_size = beam_size
        self.max_evi_num = max_evi_num
        self.causal_method = causal_method
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.mlp1 = nn.Sequential(  # 计算邻接矩阵的权重
            Linear(3*nfeat, nfeat),
            ELU(),
            Linear(nfeat, 1),
            ELU(),
        )

        self.attention = SelfAttention(nfeat*2)

        if "nwgm" in self.causal_method:
            self.linear1 = Linear(nfeat , nfeat) # 映射路径编码
            self.linear2 = Linear(nfeat , nfeat) # 映射全局图表示编码
   
            self.classifier1 = nn.Sequential(
                Linear(nfeat , nfeat),
                ELU(),
                Linear(nfeat , nclass),
                ELU(),
            )
            for m in self.classifier1.modules():
                if isinstance(m, (nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        
        if "attention" in self.causal_method:
            self.linear3 = Linear(nfeat , 64)
            self.linear4 = Linear(nfeat , 64)

            nn.init.xavier_uniform_(self.linear3.weight)
            nn.init.xavier_uniform_(self.linear4.weight)
        
        if "cf" in self.causal_method:
            self.classifier_claim = nn.Sequential(
                Linear(nfeat , nclass),
                ELU(),
            )
            self.constant = nn.Parameter(torch.tensor(0.0))
        
        self.lstm = nn.LSTM(nfeat,nfeat,2,batch_first=True)
        # self.bn = BatchNorm1d(nfeat)
        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat),
            ELU(),
            Linear(nfeat , nclass),
            ELU(),
        )
        for m in self.mlp1.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.classifier.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        # BN initialization.
        # for m in self.modules():
        #     if isinstance(m, (torch.nn.BatchNorm1d)):
        #         torch.nn.init.constant_(m.weight, 1)
        #         torch.nn.init.constant_(m.bias, 0.0001)
    
    def paths_to_class_logit(self,x,paths,logits,paths_mask,evi_labels): 
        # [batch,6,768] [batch,5,4] [batch,5] [batch,5,4] [batch,6]
        evidences, claims = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(evidences, claims, evi_labels[:,1:]) # [batch,768]

        paths = torch.LongTensor(paths).to(x.device) # [batch,5,4]
        paths_mask = torch.LongTensor(paths_mask).to(x.device) # [batch,5,4]
        logits = [[j.unsqueeze(0) for j in i] for i in logits]
        logits = [torch.cat(i,dim=0) for i in logits]
        logits = torch.cat([i.unsqueeze(0) for i in logits],dim=0) # [batch,5]
        # logits = torch.tensor(logits, device=x.device) # [batch,5] # 用这个，梯度就断啦！
        logits = F.softmax(logits,dim=1) # [batch,5]
        paths_rep = []
        for i in range(len(x)):
            graph = x[i] # [6,768]
            rep = graph[paths[i]] # [5,4,768]
            paths_rep.append(rep.unsqueeze(0))
        paths_rep = torch.cat(paths_rep,dim=0) # [batch,5,4,768]
        paths_rep = paths_rep * paths_mask.unsqueeze(-1) # [batch,5,4,768]
        shape = paths_rep.shape # [batch,5,4,768]
        paths_rep = paths_rep.view(shape[0]*shape[1],shape[2],shape[3]).contiguous() # [batch*5,4,768]

        graph_rep = graph_rep.unsqueeze(1).expand(-1,shape[1],-1) # [batch,5,768]
        graph_rep = graph_rep.contiguous().view(shape[0]*shape[1],-1) # [batch*5,768]
        graph_rep = graph_rep.unsqueeze(0).expand(2,-1,-1).contiguous() # [2,batch*5,768]

        h0, c0 = graph_rep, graph_rep
        output, (hn, cn) = self.lstm(paths_rep,(h0,c0)) # [batch*5,4,768] [1,batch*5,768]
        # output, (hn, cn) = self.lstm(paths_rep) # [batch*5,4,768] [1,batch*5,768]
        paths_rep = output[:,-1,:] # [batch*5,768]
        paths_rep = paths_rep.view(shape[0],shape[1],-1) # [batch,5,768]

        paths_class = self.classifier(paths_rep) # [batch,5,3]
        paths_class = F.softmax(paths_class,dim=-1) # [batch,5,3]
        paths_class = paths_class * logits.unsqueeze(-1) # [batch,5,3]
        paths_class = paths_class.sum(dim=1) # [batch,3]
        return paths_class # [batch,3]
    
    def claim_classifier(self,claim_batch,):  # claim_batch [batch,768]
        res = self.classifier_claim(claim_batch) # [batch,3]
        return res # [10,3]
    
    def fusion(self,res_claim,path_res):
        res_final = torch.log(1e-9 + torch.sigmoid(res_claim + path_res))
        cf_final = torch.log(1e-9 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(path_res)))
        tie = res_final - cf_final
        return res_claim, res_final, cf_final, tie
    
    def paths_to_class_logit_nwgm(self,x,paths,logits,paths_mask,evi_labels,centers):  # centers [3,768]
        # [batch,6,768] [batch,5,4] [batch,5] [batch,5,4] [batch,6] [3,768]
        paths = torch.LongTensor(paths).to(x.device) # [batch,5,4]
        paths_mask = torch.LongTensor(paths_mask).to(x.device) # [batch,5,4]
        logits = [[j.unsqueeze(0) for j in i] for i in logits]
        logits = [torch.cat(i,dim=0) for i in logits]
        logits = torch.cat([i.unsqueeze(0) for i in logits],dim=0) # [batch,5]
        # logits = torch.tensor(logits, device=x.device) # [batch,5] # 用这个，梯度就断啦！
        logits = F.softmax(logits,dim=1) # [batch,5]
        paths_rep = []
        for i in range(len(x)):
            graph = x[i] # [6,768]
            rep = graph[paths[i]] # [5,4,768]
            paths_rep.append(rep.unsqueeze(0))
        paths_rep = torch.cat(paths_rep,dim=0) # [batch,5,4,768]
        paths_rep = paths_rep * paths_mask.unsqueeze(-1) # [batch,5,4,768]
        shape = paths_rep.shape # [batch,5,4,768]
        paths_rep = paths_rep.view(shape[0]*shape[1],shape[2],shape[3]).contiguous() # [batch*5,4,768]

        # 当前样本的图表示
        evidences, claims = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(evidences, claims, evi_labels[:,1:]) # [batch,768]
        graph_rep = graph_rep.unsqueeze(1).expand(-1,shape[1],-1) # [batch,5,768]
        graph_rep = graph_rep.contiguous().view(shape[0]*shape[1],-1) # [batch*5,768]
        graph_rep = graph_rep.unsqueeze(0) # [1,batch*5,768]
        graph_rep = torch.cat([graph_rep, graph_rep],dim=0) # [2,batch*5,768]

        h0, c0 = graph_rep, graph_rep
        output, (hn, cn) = self.lstm(paths_rep,(h0,c0)) # [batch*5,4,768] [1,batch*5,768]
        paths_rep = output[:,-1,:] # [batch*5,768]
        paths_rep = paths_rep.view(shape[0],shape[1],-1) # [batch,5,768]

        paths_class = self.classifier1(paths_rep.detach()) # [batch,5,3]
        paths_class = F.softmax(paths_class,dim=-1) # [batch,5,3]
        # 整个数据集的图表示加权和
        graph_rep1 = paths_class.unsqueeze(-1) * centers # [batch,5,3,768]
        graph_rep1 = graph_rep1.mean(dim=2) # [batch,5,768]
        # graph_rep1 = graph_rep1.view(shape[0]*shape[1],-1) # [batch*5,768]
        # graph_rep1 = graph_rep1.unsqueeze(0) # [1,batch*5,768]

        # inputs = torch.cat([paths_rep,graph_rep1],dim=-1) # [batch,5,768*2]
        inputs = self.linear1(paths_rep) + 0.1 * self.linear2(graph_rep1) # [batch,5,768]
        # inputs = paths_rep + 0.1 * graph_rep1  # [batch,5,768]
        # paths_class = self.linear1(paths_rep) + 0.01 * self.linear2(graph_rep1) # [batch,5,3]

        # graph_rep1 = torch.cat([graph_rep, graph_rep1],dim=0) # [2,batch*5,768]

        # h0, c0 = graph_rep1, graph_rep1
        # output, (hn, cn) = self.lstm2(paths_rep,(h0,c0)) # [batch*5,4,768] [1,batch*5,768]
        # paths_rep2 = output[:,-1,:] # [batch*5,768]
        # paths_rep2 = paths_rep2.view(shape[0],shape[1],-1) # [batch,5,768]

        # inputs = inputs.view(-1,shape[-1]) # [batch*5,768]

        # inputs = self.bn(inputs)

        # inputs = inputs.view(shape[0],shape[1],shape[-1])

        paths_class = self.classifier(inputs) # [batch,5,3]
        paths_class = F.softmax(paths_class,dim=-1) # [batch,5,3]

        paths_class = paths_class * logits.unsqueeze(-1) # [batch,5,3]
        paths_class = paths_class.sum(dim=1) # [batch,3]

        return paths_class # [10,3]
    
    def paths_to_class_logit_nwgm_attention(self,x,paths,logits,paths_mask,evi_labels,centers):  # centers [3,n_cluster,768]
        # [batch,6,768] [batch,5,4] [batch,5] [batch,5,4] [batch,6] [3,768]
        paths = torch.LongTensor(paths).to(x.device) # [batch,5,4]
        paths_mask = torch.LongTensor(paths_mask).to(x.device) # [batch,5,4]
        logits = [[j.unsqueeze(0) for j in i] for i in logits]
        logits = [torch.cat(i,dim=0) for i in logits]
        logits = torch.cat([i.unsqueeze(0) for i in logits],dim=0) # [batch,5]
        # logits = torch.tensor(logits, device=x.device) # [batch,5] # 用这个，梯度就断啦！
        logits = F.softmax(logits,dim=1) # [batch,5]
        paths_rep = []
        for i in range(len(x)):
            graph = x[i] # [6,768]
            rep = graph[paths[i]] # [5,4,768]
            paths_rep.append(rep.unsqueeze(0))
        paths_rep = torch.cat(paths_rep,dim=0) # [batch,5,4,768]
        paths_rep = paths_rep * paths_mask.unsqueeze(-1) # [batch,5,4,768]
        shape = paths_rep.shape # [batch,5,4,768]
        paths_rep = paths_rep.view(shape[0]*shape[1],shape[2],shape[3]).contiguous() # [batch*5,4,768]

        # 当前样本的图表示
        evidences, claims = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(evidences, claims, evi_labels[:,1:]) # [batch,768]
        graph_rep = graph_rep.unsqueeze(1).expand(-1,shape[1],-1) # [batch,5,768]
        graph_rep = graph_rep.contiguous().view(shape[0]*shape[1],-1) # [batch*5,768]
        graph_rep = graph_rep.unsqueeze(0) # [1,batch*5,768]
        graph_rep = torch.cat([graph_rep, graph_rep],dim=0) # [2,batch*5,768]

        h0, c0 = graph_rep, graph_rep
        output, (hn, cn) = self.lstm(paths_rep,(h0,c0)) # [batch*5,4,768] [1,batch*5,768]
        paths_rep = output[:,-1,:] # [batch*5,768]
        paths_rep = paths_rep.view(shape[0],shape[1],-1) # [batch,5,768]

        q = self.linear3(paths_rep) # [batch,5,256]
        q = q.view(shape[0]*shape[1],-1) # [batch*5,256]
        q = q.unsqueeze(1).unsqueeze(1) # [batch*5,1,1,256]
        k = self.linear4(centers) # [3,n_cluster,256]
        graph_rep_list = []
        for i in range(len(k)):
            k_i = k[i] # [n_cluster,256]
            k_i = k_i.unsqueeze(-1) # [n_cluster,256,1]
            v = centers[i] # [n_cluster,768]
            weight = torch.matmul(q,k_i) # [batch*5,n_cluster,1,1]
            # weight = weight / torch.sqrt(torch.tensor(64))
            weight = weight.squeeze(-1) # [batch*5,n_cluster,1]
            weight = weight.transpose(1,2) # [batch*5,1,n_cluster]
            weight = F.softmax(weight,dim=-1) # [batch*5,1,n_cluster]
            res = torch.matmul(weight,v) # [batch*5,1,768]
            graph_rep_list.append(res)
        graph_rep_list = torch.cat(graph_rep_list,dim=1) # [batch*5,3,768]
        graph_rep_list = graph_rep_list.view(shape[0],shape[1],len(k),-1) # [batch,5,3,768]

        paths_class1 = self.classifier1(paths_rep) # [batch,5,3]
        paths_class1 = F.softmax(paths_class1,dim=-1) # [batch,5,3]
        # 整个数据集的图表示加权和
        graph_rep1 = paths_class1.unsqueeze(-1) * graph_rep_list # [batch,5,3,768]
        graph_rep1 = graph_rep1.mean(dim=2) # [batch,5,768]

        inputs = self.linear1(paths_rep) + 0.1 * self.linear2(graph_rep1) # [batch,5,768]

        paths_class = self.classifier(inputs) # [batch,5,3]
        paths_class = F.softmax(paths_class,dim=-1) # [batch,5,3]

        paths_class = paths_class * logits.unsqueeze(-1) # [batch,5,3]
        paths_class = paths_class.sum(dim=1) # [batch,3]

        paths_class1 = paths_class1 * logits.unsqueeze(-1) # [batch,5,3]
        paths_class1 = paths_class1.sum(dim=1) # [batch,3]

        return paths_class, paths_class1 # [10,3]
    
    def forward(self, data, centers=None, evi_supervision=False):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.max_evi_num,pooled_output.shape[-1]) # [batch,6,768]
        # pooled_output = F.normalize(pooled_output,dim=-1) # 归一化
        feature_batch, claim_batch = pooled_output[:,1:,:], pooled_output[:,0,:] # [batch,5,768] # [batch,768]

        datas = []
        for i in range(len(feature_batch)):
            x = torch.cat([claim_batch[i].unsqueeze(0),
                           feature_batch[i]],dim=0) # [6,768]
            # 全连接
            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            # data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        x, edge_index = datas.x, datas.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)

        x = x.view(-1,1+self.max_evi_num,x.shape[-1]) # [batch,6,768]
        # x = torch.cat([claim_batch.unsqueeze(1),feature_batch],dim=1) # [batch,6,768]
        probability = self.cal_transition_probability_matrix(x,sent_labels)  # [batch,6,6]
        # feature_batch, claim_batch = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]

        paths, logits, paths_mask = self.batch_find_path(probability, self.max_length, self.beam_size)
        if centers is not None and self.causal_method == "kmeans":
            res = self.paths_to_class_logit_kmeans(x,paths,logits,paths_mask,centers,center_logits)
        elif centers is not None and self.causal_method == "nwgm":
            res = self.paths_to_class_logit_nwgm(x,paths,logits,paths_mask,sent_labels,centers)
        elif self.causal_method == "nwgm_cf":
            res = self.paths_to_class_logit_nwgm(x,paths,logits,paths_mask,sent_labels,centers)
            res_claim = self.claim_classifier(claim_batch)
            res_claim, res_final, cf_final, tie = self.fusion(res_claim,res)
        elif "attention" in self.causal_method:
            res, res1 = self.paths_to_class_logit_nwgm_attention(x, paths, logits, paths_mask, evi_labels, centers)
        else:
            res = self.paths_to_class_logit(x,paths,logits,paths_mask,sent_labels)

        if evi_supervision and self.causal_method=="nwgm_cf":
            return res_claim, res_final, cf_final, tie, probability
        elif evi_supervision and "attention" in self.causal_method:
            return res, res1, probability
        elif evi_supervision:
            return res, probability
        else:
            return res # [batch,3]
    # 计算转移概率矩阵
    def cal_transition_probability_matrix(self, x, evi_labels): # [batch,6,768] [batch,6]
        node_num = x.shape[1]
        mat_rep = torch.cat([x.unsqueeze(2).expand(-1,-1,node_num,-1), #[batch,6,6,768]
                             x.unsqueeze(1).expand(-1,node_num,-1,-1), #[batch,6,6,768]
                             x[:,0,:].unsqueeze(1).unsqueeze(1).expand(-1,node_num,node_num,-1), #[batch,6,6,768]
                             ],dim=-1) # [batch,6,6,768*3]
        weight = self.mlp1(mat_rep) # [batch,6,6,1]
        weight = weight.squeeze(-1) # [batch,6,6]
        
        # weight = torch.matmul(x,x.transpose(1,2)) # [batch,6,6]
        # print(weight.sum().item())
        evi_labels = evi_labels.to(torch.float)
        mask = torch.matmul(evi_labels.unsqueeze(2),evi_labels.unsqueeze(1)) # [batch,6,6]
        mask = mask + torch.diag_embed(torch.ones(mask.shape[1])).to(mask.device) # 保证每行都至少有一个数
        mask = mask == 0
        mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) # 邻接矩阵中为0的地方填上负无穷 [batch,6,6]
        weight = weight + mask
        probability = F.softmax(weight,dim=-1) # [batch,6,6]
        return probability # [batch,6,6]
    
    # 一个样本寻找路径
    def find_path(self, start_node, probability, max_length, beam_size): # 0 [6,6] 4 5
        top_beam_paths = [[start_node]] # 概率最高的beam_size个路径
        top_beam_logits = [torch.tensor(0.0).to(probability.device)] # 每条路径的概率 [5]
        top_beam_end = [0] # 这个列表表示对应路径是否已经终止搜索
        while not all(top_beam_end):
            new_paths = []
            new_logits = []
            for k in range(len(top_beam_paths)):
                if top_beam_end[k]: # 已终止搜索
                    continue
                path = top_beam_paths[k]
                logit = top_beam_logits[k]
                curr_node = path[-1]
                edge = probability[curr_node] # [6]

                next_nodes = []
                values = []
                for i, value in enumerate(edge):
                    if i not in path and value > 0.0:
                        next_nodes.append(i)
                        values.append(value)
                top_index = heapq.nlargest(min(beam_size,len(values)), range(len(values)), values.__getitem__)
                next_nodes = [next_nodes[j] for j in top_index] 
                values = [values[j] for j in top_index]
                
                new_paths += [path + [j] for j in next_nodes]
                new_logits += [logit + torch.log(j+1e-8) for j in values]

            # 把新解码得到的路径和已经终止搜索的路径合并起来
            all_paths = []
            all_logits = []
            for index, end in enumerate(top_beam_end):
                if end:
                    all_paths.append(top_beam_paths[index])
                    all_logits.append(top_beam_logits[index])
            all_paths += new_paths
            all_logits += new_logits

            # 取top beam_size个
            temp_logits = [logit/len(all_paths[index]) for index, logit in enumerate(all_logits)]
            top_index = heapq.nlargest(min(beam_size,len(temp_logits)), range(len(temp_logits)), temp_logits.__getitem__)
            top_beam_paths = [all_paths[j] for j in top_index] 
            top_beam_logits = [all_logits[j] for j in top_index]

            # 判断取到的路径是否终止搜索
            top_beam_end = []
            for i in range(len(top_beam_paths)):
                end = 0
                if len(top_beam_paths[i]) >= max_length:
                    end = 1
                curr_node = top_beam_paths[i][-1]
                edge = probability[curr_node]
                next_nodes = []
                for j, value in enumerate(edge):
                    if j not in path and value > 0.0:
                        next_nodes.append(j)
                top_index = heapq.nlargest(min(beam_size,len(values)), range(len(values)), values.__getitem__)
                next_nodes = [next_nodes[j] for j in top_index] 
                if all([node in top_beam_paths[i] for node in next_nodes]) or len(next_nodes)==0:
                    end = 1
                top_beam_end.append(end)
        
        # 归一化
        top_beam_logits = [logit/len(top_beam_paths[index]) for index, logit in enumerate(top_beam_logits)]
        paths = top_beam_paths
        logits = top_beam_logits
        # print([test.item() for test in logits])

        return paths, logits # [5,4] [5]
    
    def batch_find_path(self, probability, max_length, beam_size): # [batch,6,6]
        all_paths = [] # [batch,5,4]
        all_paths_mask = [] # [batch,5,4] 表示路径上的结点是否存在
        all_logits = [] # [batch,5]
        for i in range(len(probability)):
            paths, logits = self.find_path(0,probability[i],max_length,beam_size)
            paths_mask = []
            for j in range(len(paths)):
                # 将路径长度补到最大
                paths_mask.append([1]*len(paths[j])+[0]*(max_length - len(paths[j])))
                paths[j] = paths[j] + [0] * (max_length - len(paths[j]))
            # 补路径数量
            paths_mask += [[0]*max_length]*(beam_size-len(paths))
            logits += [torch.tensor(float("-inf")).to(probability.device)]*(beam_size-len(paths))
            paths += [[0]*max_length]*(beam_size-len(paths))
            
            all_paths.append(paths)
            all_paths_mask.append(paths_mask)
            all_logits.append(logits)
        return all_paths, all_logits, all_paths_mask # [batch,5,4] [batch,5] [batch,5,4]
    

class CrossAttention(nn.Module):
    def __init__(self, nhid):
        super(CrossAttention, self).__init__()
        self.project_c = Linear(nhid, 64)
        self.project_e = Linear(nhid, 64)

        self.f_align = Linear(4*nhid,nhid)
    def forward(self, x, datas):

        batch = datas.batch
        # res = []
        # for i in range(datas.num_graphs):
        #     example_index = (batch == i).nonzero().squeeze()
        #     data = datas.get_example(i)
        #     claim = x[example_index][data.claim_index] # [20,768]
        #     evidence = x[example_index][data.evidence_index] # [100,768]
        #     weight_c = self.project_c(claim)  # [20,64]
        #     weight_e = self.project_e(evidence)  # [100,64]
        #     weight = torch.matmul(weight_c, weight_e.transpose(0,1)) # [20,100]
        #     weight = F.softmax(weight,dim=-1) # [20,100]
        #     claim_new = torch.matmul(weight,evidence) # [20,768]

        #     a = torch.cat([claim,claim_new,claim-claim_new,claim*claim_new],dim=-1) # [20,768*4]
        #     a = self.f_align(a) # [20,768]
        #     a = a.mean(dim=0) # [768]
        #     res.append(a.unsqueeze(0))
        # res = torch.cat(res,dim=0) # [128,768]

        claim_batch = batch[datas.claim_index] # [500]
        evidence_batch = batch[datas.evidence_index] # [1000]

        mask = ~(claim_batch.unsqueeze(1) == evidence_batch.unsqueeze(0))  # [500,1000]
        mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) # [500,1000]
        claim = x[datas.claim_index] # [500,768]
        evidence = x[datas.evidence_index] # [1000,768]
        weight_c = self.project_c(claim)  # [500,64]
        weight_e = self.project_e(evidence)  # [1000,64]
        weight = torch.matmul(weight_c, weight_e.transpose(0,1)) # [500,1000]
        weight = weight + mask
        weight = F.softmax(weight,dim=-1) # [500,1000]
        claim_new = torch.matmul(weight,evidence) # [500,768]

        a = torch.cat([claim,claim_new,claim-claim_new,claim*claim_new],dim=-1) # [500,768*4]
        a = self.f_align(a) # [500,768]
        res = global_mean_pool(a, claim_batch) # [128,768]
        
        return res


class CLASSIFIER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLASSIFIER, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp = nn.Sequential(  # 三分类
            Linear(nfeat, nclass),
            ELU(True),
        )

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        res = self.mlp(pooled_output)
        return res

class CLEVER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLEVER, self).__init__()
        self.bert1 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.bert2 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp1 = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.mlp2 = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert1(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, evidences = self.bert2(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        # res_claim = torch.log(1e-8 + self.mlp1(claims)) # [batch,3]
        # res_fusion = torch.log(1e-8 + self.mlp2(evidences)) # [batch,3]
        res_claim = self.mlp1(claims) # [batch,3]
        res_fusion = self.mlp2(evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie


class CICR(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CICR, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.classifier_claim = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.classifier_evidence = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_evidence = torch.nn.Parameter(torch.zeros((nfeat)))
        self.classifier_fusion = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        # self.linear1 = Linear(nclass, nclass)
        # self.linear2 = Linear(nclass, nclass)
      

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, claim_evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]

        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,2,:] # [batch,128]
        input_mask = input_mask[:,2,:] # [batch,128]
        segment_ids = segment_ids[:,2,:] # [batch,128]
        _, evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        claims = claims.detach()
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.classifier_evidence(evidences) # [batch,3]
        res_fusion = self.classifier_fusion(claim_evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE


class CLEVER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLEVER, self).__init__()
        self.bert1 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.bert2 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp1 = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.mlp2 = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert1(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, evidences = self.bert2(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        # res_claim = torch.log(1e-8 + self.mlp1(claims)) # [batch,3]
        # res_fusion = torch.log(1e-8 + self.mlp2(evidences)) # [batch,3]
        res_claim = self.mlp1(claims) # [batch,3]
        res_fusion = self.mlp2(evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie

class CLEVER_graph(nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num):
        super(CLEVER_graph, self).__init__()
        self.evi_max_num = evi_max_num
        self.fusion_model = ONE_ATTENTION_with_bert(nfeat, nclass, evi_max_num)
        self.claim_model = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp_claim = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        res_fusion = self.fusion_model(data) # [batch,3]

        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels, indexs = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.claim_model(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        res_claim = self.mlp_claim(claims) # [batch,3]
   
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie

class ONE_ATTENTION(nn.Module): # 不带bert
    def __init__(self, nfeat, nclass, evi_max_num, pool):
        super(ONE_ATTENTION, self).__init__()
        self.evi_max_num = evi_max_num
        self.pool = pool
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.attention = SelfAttention(nfeat*2)
        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat),
            ELU(True),
            Linear(nfeat, nclass),
            ELU(True),
        )
    
    def forward(self, pooled_output, sent_labels): # [batch,6,768]
        datas = []
        for i in range(len(pooled_output)):
            x = pooled_output[i] # [6,768]
            # 全连接
            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        x, edge_index = datas.x, datas.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)
        
        if self.pool == "att":
            x = x.view(-1,1+self.evi_max_num,x.shape[-1]) # [batch,6,768]
            feature_batch, claim_batch = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
            graph_rep = self.attention(feature_batch, claim_batch, sent_labels[:,1:]) # [batch,768]
        else:
            x = x.view(-1,self.evi_max_num,x.shape[-1]) # [batch,6,768]
            graph_rep = x.mean(dim=1) # [batch,768]

        outputs = self.classifier(graph_rep)
        return outputs

class CICR_graph(nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num):
        super(CICR_graph, self).__init__()
        self.evi_max_num = evi_max_num
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.evidence_model = ONE_ATTENTION(nfeat, nclass, evi_max_num, "mean")
        self.fusion_model = ONE_ATTENTION(nfeat, nclass, evi_max_num, "att")
        self.classifier_claim = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.D_u = torch.randn((nclass,nfeat))
        self.linear1 = Linear(nfeat,64)
        self.linear2 = Linear(nfeat,64)
        self.linear3 = Linear(nfeat,nfeat)
    
    def claim_intervention(self, claims): # [batch,768]
        D_u = self.D_u.to(claims.device) # [3,768]
        L = self.linear1(claims) # [batch,64]
        K = self.linear2(D_u) # [3,64]
        w = torch.matmul(L,K.transpose(0,1)) # [batch,3]
        w = F.softmax(w,dim=-1)
        E_D_u = torch.matmul(w,D_u) # [batch,768]
        claims = self.linear3(claims + E_D_u)
        return claims

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels, indexs = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.evi_max_num,pooled_output.shape[-1]) # [batch,1+5,768]
        claims = pooled_output[:,0,:] # [batch,768]

        claim_evidences = pooled_output # [batch,6,768]

        evidences = pooled_output[:,1:,:] # [batch,5,768]
      
        claims = claims.detach()
        claims = self.claim_intervention(claims)
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.evidence_model(evidences,sent_labels[:,1:]) # [batch,3]
        res_fusion = self.fusion_model(claim_evidences,sent_labels) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE


class CICR(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CICR, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.classifier_claim = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.classifier_evidence = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_evidence = torch.nn.Parameter(torch.zeros((nfeat)))
        self.classifier_fusion = nn.Sequential(  # 三分类
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.D_u = torch.randn((nclass,nfeat))
        self.linear1 = Linear(nfeat,64)
        self.linear2 = Linear(nfeat,64)
        self.linear3 = Linear(nfeat,nfeat)

    
    def claim_intervention(self, claims): # [batch,768]
        D_u = self.D_u.to(claims.device) # [3,768]
        L = self.linear1(claims) # [batch,64]
        K = self.linear2(D_u) # [3,64]
        w = torch.matmul(L,K.transpose(0,1)) # [batch,3]
        w = F.softmax(w,dim=-1)
        E_D_u = torch.matmul(w,D_u) # [batch,768]
        claims = self.linear3(claims + E_D_u)
        return claims
      

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, claim_evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]

        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,2,:] # [batch,128]
        input_mask = input_mask[:,2,:] # [batch,128]
        segment_ids = segment_ids[:,2,:] # [batch,128]
        _, evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        claims = claims.detach()
        claims = self.claim_intervention(claims)
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.classifier_evidence(evidences) # [batch,3]
        res_fusion = self.classifier_fusion(claim_evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE