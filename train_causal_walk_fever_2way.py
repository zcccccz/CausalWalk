import random, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import texar.torch as tx
from pathlib import Path
from shutil import copyfile
import json

from utils import init_logger, load_pair_data, cal_nwgm_center_attention
from data_load_utils import build_dataset
from models import Walk_with_bert
from torch.cuda.amp import autocast, GradScaler
from pytorch_pretrained_bert.optimization import BertAdam

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
parser.add_argument("--max_seq_length", default=128, type=int,)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

dir_path = Path('outputs/causal_walk_fever_2way/')

train_data_path = "data/fever/fever_train-2class-mh.json"
dev_file_paths = [
    "data/fever/fever_dev-2class.json",
    "data/fever/fever_dev-2class-mh.json",
    "data/fever/fever2-2class.json",
    "data/fever/fever2-2class-mh.json"
    ]
graph_rep_model_path = "pretrained_models/graph_rep_model_fever.pt"
nclass = 2
max_length = 4
beam_size = 3
causal_method = "nwgm_attention"
n_cluster = 10
opt_steps = 1

tx.utils.maybe_create_dir(dir_path)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
if os.path.exists(dir_path / 'results.json'):
    print(dir_path / 'results already exists!')
    exit(0)
else:
    print(dir_path)
# 将当前实验的代码复制到日志文件夹
# tx.utils.maybe_create_dir(dir_path / 'code')
# for i in Path().iterdir():
#     if i.is_file():
#         copyfile(i,dir_path / 'code' / i)
logger = init_logger(dir_path / "log.log")
tx.utils.maybe_create_dir(dir_path / 'models')

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step):
    model.eval()
    for path in dev_data_list:
        dev_data = dev_data_list[path]
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        dev_tqdm_iterator = dev_dataloader
        correct_pred = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for index, data in enumerate(dev_tqdm_iterator):
                data = [i.to(device) for i in data]
                input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data

                outputs, outputs1, probability = model(data, train_centers, True)  # probability [batch,6,6]
                loss = F.cross_entropy(outputs, labels)
                loss1 = F.cross_entropy(outputs1, labels)
                loss = loss + loss1
                # loss = loss_func(outputs, labels)

                # mask = sent_labels == 0
                # mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf"))
                # evi_labels = evi_labels + mask
                # evi_labels = F.softmax(evi_labels,dim=-1)
                # evi_labels = evi_labels.unsqueeze(1) # [batch,1,6]

                # evi_labels = torch.log(1e-9 + evi_labels) # [batch,1,6]
                # probability = torch.log(1e-9 + probability) # [batch,6,6]

                # kl_loss = F.kl_div(probability, evi_labels, log_target=True)

                # loss = loss + 0.1 * kl_loss

                correct = correct_prediction(outputs, labels)
                correct_pred += correct
                running_loss += loss.item()
        
        dev_loss = running_loss / len(dev_dataloader)
        dev_accuracy = correct_pred / len(dev_data)
        logger.info('%s Dev total acc: %lf, total loss: %lf\r\n' % (path, dev_accuracy, dev_loss))

        accuracy[path].append(dev_accuracy.item())

        if dev_accuracy > best_accuracy[path]:
            best_accuracy[path] = dev_accuracy.item()
            best_step[path] = step
            torch.save({'model': model.state_dict(),
                        'best_accuracy': best_accuracy[path],
                        'dev_losses': dev_loss},
                        '%s/models/%s.pt' % (dir_path,path.split("/")[-1]))
    model.train()
    return best_accuracy, accuracy, best_step

logger.info("load data...")
train_data = build_dataset(train_data_path, args.evi_num, args.max_seq_length)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
total_steps = len(train_dataloader) * args.epochs / opt_steps

feature_num = 768
dev_data_list = {}
best_accuracy = {}
best_step = {}
accuracy = {}
dev_centers_list = {}
for path in dev_file_paths:
    dev_data = build_dataset(path, args.evi_num, args.max_seq_length)
    dev_data_list[path] = dev_data
    best_accuracy[path] = 0.0
    accuracy[path] = []
    best_step[path] = 0
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    dev_dataloader = tqdm(dev_dataloader)
    dev_centers_list[path] = cal_nwgm_center_attention(graph_rep_model_path, dev_dataloader, nclass, device, feature_num, args.evi_num, n_cluster)

feature_num = 768
model = Walk_with_bert(feature_num, nclass, max_length, beam_size, args.evi_num, causal_method)
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
t_total = len(train_dataloader) * args.epochs / opt_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.lr,
                        warmup=0.1,
                        t_total=t_total)

logger.info("start train...")
train_tqdm_iterator = tqdm(train_dataloader)
train_centers = cal_nwgm_center_attention(graph_rep_model_path, train_tqdm_iterator, nclass, device, feature_num, args.evi_num, n_cluster)
step = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    correct_pred = 0.0
    train_tqdm_iterator = tqdm(train_dataloader)
    for index, data in enumerate(train_tqdm_iterator):
        data = [i.to(device) for i in data]
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data  # sent_labels [batch,6] evi_labels [batch,6]
    
        outputs, outputs1, probability = model(data, train_centers, True)  # probability [batch,6,6]
        loss = F.cross_entropy(outputs, labels)
        loss1 = F.cross_entropy(outputs1, labels)
        loss = loss + loss1
        # loss = loss_func(outputs, labels)

        # mask = sent_labels == 0
        # mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf"))
        # evi_labels = evi_labels + mask
        # evi_labels = F.softmax(evi_labels,dim=-1)
        # evi_labels = evi_labels.unsqueeze(1) # [batch,1,6]

        # evi_labels = torch.log(1e-9 + evi_labels) # [batch,1,6]
        # probability = torch.log(1e-9 + probability) # [batch,6,6]

        # kl_loss = F.kl_div(probability, evi_labels, log_target=True)

        # loss = loss + 0.1 * kl_loss

        correct = correct_prediction(outputs, labels)
        correct_pred += correct
        running_loss += loss.item()

        description = 'epoch %d Acc: %lf, Loss: %lf' % (epoch, correct_pred / (index + 1) / args.batch_size, running_loss / (index + 1))
        train_tqdm_iterator.set_description(description)

        loss = loss / opt_steps
        loss.backward()
        if index % opt_steps == 0 or index+1 == len(train_tqdm_iterator):
            optimizer.step()
            optimizer.zero_grad()
            # best_accuracy, accuracy, best_step = eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step)
            step += 1

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_pred / len(train_data)
    logger.info('epoch: %d, Train acc: %lf, total loss: %lf\r\n' % (epoch, train_accuracy, train_loss))
    best_accuracy, accuracy, best_step = eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step)

logger.info(json.dumps(best_accuracy,indent=True))
logger.info(json.dumps(best_step,indent=True))
logger.info("total_steps: %d"%total_steps)

res = {
    "total_steps":total_steps,
    "best_accuracy":best_accuracy,
    "best_step":best_step,
    "accuracy":accuracy,
}
fout = open(dir_path / 'results.json', 'w')
fout.write(json.dumps(res,indent=True))
fout.close()