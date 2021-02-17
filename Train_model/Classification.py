#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os, pickle
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
import logging

# In[14]:


class BreathData_cycle(Dataset):
    def __init__(self, data_path, mode="train", val_ratio=0.2, test_ratio=0.2):
        with open(data_path, "rb") as file:
            data_dict = pickle.load(file)
        self.mode = mode
        all_data, all_labels = [],[]
        self.people = [i for i in data_dict.keys()]
        self.label_mapping = {self.people[i]:i for i in range(len(self.people))}
        label2person = {self.label_mapping[person] for person in self.label_mapping.keys()}
        self.save_mapping = {"person2label":self.label_mapping, "label2person":label2person}
        for person, this_data in data_dict.items():
            this_cycle = len(this_data)
            for each in this_data:
                print(each.shape)
            all_data += this_data
            all_labels += [self.label_mapping[person]] * this_cycle
        all_labels = np.asarray(all_labels)
        
        num_data = len(all_labels)
        train_ratio = 1 - val_ratio - test_ratio
        np.random.seed(0)
        random_idx = np.random.permutation(num_data)
        train_end_idx, val_end_idx = round(num_data * train_ratio), round(num_data * (train_ratio+val_ratio))
        train_idx, val_idx = random_idx[:train_end_idx], random_idx[train_end_idx:val_end_idx]
        if test_ratio != 0.0:
            test_idx = random_idx[val_end_idx:]
        if self.mode=="train":
            self.data = [all_data[i] for i in train_idx]
            self.labels = all_labels[train_idx]
        elif self.mode=="val":
            self.data = [all_data[i] for i in val_idx]
            self.labels = all_labels[val_idx]
        elif self.mode=="test":
            self.data = [all_data[i] for i in test_idx]
            self.labels = all_labels[test_idx]

        print(self.mode)
        #for each in self.data:
        #    print(each.shape)
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self,item):
        return self.data[item], self.labels[item]

class BreathData_cycle2(Dataset):
    def __init__(self, data_path, mode="train", val_ratio=0.2, test_ratio=0.2):
        with open(data_path, "rb") as file:
            data_dict = pickle.load(file)
        self.mode = mode
        train_data, train_labels, val_data, val_labels, test_data, test_labels = [],[],[],[],[],[]
        self.people = [i for i in data_dict.keys()]
        self.label_mapping = {self.people[i]:i for i in range(len(self.people))}
        label2person = {self.label_mapping[person]:person for person in self.label_mapping.keys()}
        self.save_mapping = {"person2label":self.label_mapping, "label2person":label2person}
        self.idx_record, self.idx_start = {"train":{}, "val":{}, "test":{}}, {"train":{}, "val":{}, "test":{}}
        idx_start_train, idx_start_val, idx_start_test = 0,0,0
        for itid, (person, this_data) in enumerate(data_dict.items()):
        
            this_cycle = len(this_data)
            cut_idx_train = int(np.round(this_cycle * (1 - val_ratio - test_ratio)))
            cut_idx_val = int(np.round(this_cycle * (1 - test_ratio)))
            num_val, num_test = cut_idx_val - cut_idx_train, this_cycle - cut_idx_val
            np.random.seed(itid*2)
            rand_idx = np.random.permutation(this_cycle)
            train_idx, val_idx, test_idx = rand_idx[:cut_idx_train], rand_idx[cut_idx_train:cut_idx_val], rand_idx[cut_idx_val:]
            self.idx_record["train"][person] = train_idx
            self.idx_record["val"][person] = val_idx
            self.idx_record["test"][person] = test_idx

            train_data += [this_data[i] for i in train_idx]
            train_labels += [self.label_mapping[person]] * cut_idx_train
            val_data += [this_data[i] for i in val_idx]
            val_labels += [self.label_mapping[person]] * (cut_idx_val - cut_idx_train)
            test_data += [this_data[i] for i in test_idx]
            test_labels += [self.label_mapping[person]] * (this_cycle - cut_idx_val)

            self.idx_start["train"][person] = (idx_start_train, idx_start_train+cut_idx_train)
            self.idx_start["val"][person] = (idx_start_val, idx_start_val+num_val)
            self.idx_start["test"][person] = (idx_start_test, idx_start_test+num_test)

        if self.mode=="train":
            self.data = train_data
            self.labels = train_labels
        elif self.mode=="val":
            self.data = val_data
            self.labels = val_labels
        elif self.mode=="test":
            self.data = test_data
            self.labels = test_labels

        print(self.mode)

        #for each in self.data:
        #    print(each.shape)
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self,item):
        return self.data[item], self.labels[item]


class RNN_model(nn.Module):
    def __init__(self, num_classes):
        super(RNN_model, self).__init__()
        self.project_layer = nn.Linear(39, 128)
        self.rnn_layer = nn.LSTM(128,128, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        rnn_outputs, (hn, cn) = self.rnn_layer(self.project_layer(x))
        logits = self.classifier(rnn_outputs[:,-1])
        return logits


# In[4]:
def initialize_weights(model):
    if isinstance(model, nn.LSTM):
        for name, param in model.named_parameters(): 
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
      




def run_epoch(data_loader, epoch, model, criterion, device, use_cuda=True, mode = 'train', 
    optimizer=None, writer=None, idx_record=None):
    if mode == "train":
        model.train()
    elif mode in ['val', 'test']:
        model.eval()
    else:
        print("Wrong mode input!")
        return
    
    model.to(device)
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    avg_loss, avg_acc = 0.0, 0.0
    batches = 0
    output_all, label_all = [],[]
    for batch_idx, sample in enumerate(data_loader):
        data, labels = sample[0], sample[1]
        label_all.append(labels.numpy())
        data = data.type(dtype=FloatTensor).to(device)
        labels = labels.type(dtype=LongTensor).to(device)
        #print(data.size())
        #print(labels.size())
        output = model(data)
        output_class = torch.argmax(output, dim=1)
        output_all.append(output_class.cpu().numpy())
        loss = criterion(output, labels)
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss += loss.item()
        avg_acc += (output_class == labels).sum().item()
        batches += labels.size()[0]
    avg_loss /= batches
    avg_acc /= batches
    print("all data num:{}".format(batches))
    output_all, label_all = np.concatenate(output_all), np.concatenate(label_all)

    if writer is not None:
        writer.add_scalar(f"Loss/{mode}", avg_loss, epoch)
        writer.add_scalar(f"Acc/{mode}", avg_acc, epoch)

    print('%s Epoch: %d  , Loss: %.4f,  Accuracy: %.2f'%( mode, epoch, avg_loss, avg_acc))


    return avg_acc, output_all, label_all
        
def get_fpr(output_all, label_all, label2person):
    person_all = np.unique(label_all)
    fpr_each = {}
    mis_cls_all, total = 0,0
    for person in person_all:
        corr_output = output_all[label_all!=person]
        num_other = len(corr_output)
        mis_cls = (corr_output==person).sum()
        mis_cls_all += mis_cls
        total += num_other
        fpr_each[label2person[person]] = round(mis_cls / num_other)

    fpr = mis_cls_all / total
    
    return fpr, fpr_each

# In[17]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for user identity classification')
    parser.add_argument('--base_path', type=str, default="/home/qiaomu/codes/CSE518")
    parser.add_argument('--ckpt_path', type=str, default="/data/add_disk0/qiaomu/ckpts/CSE518")
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--ckpt_name', type=str, default="best_model.pt")

    args = parser.parse_args()
    data_path = os.path.join(args.base_path, "extracted_features_new.pickle")
    train_dataset = BreathData_cycle2(data_path, mode="train")
    val_dataset = BreathData_cycle2(data_path, mode="val")
    save_mapping = train_dataset.save_mapping
    label2person = save_mapping["label2person"]
    target_names = [label2person[key] for key in label2person.keys()]
    num_people = len(train_dataset.people)
    print(f"num_people:{num_people}")
    #print("train dataset indices: {}".format(train_dataset.idx_record))
    #print("val dataset indices: {}".format(val_dataset.idx_record))

    writer = SummaryWriter('./logs')
    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= args.batch_size, num_workers=4, shuffle=False)
    model = RNN_model(num_people)
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    print('use cuda: %s'%(use_cuda))
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if args.optim=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="Adam":
        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=="RMSprop":
        optimizer = optim.RMSprop(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    if args.mode=="train":
        for epoch in range(args.epochs):
            acc_train, output_train, label_train = run_epoch(train_loader, epoch, model, criterion, device, use_cuda, mode='train', optimizer=optimizer, writer=writer)
            acc_val, output_val, label_val = run_epoch(val_loader, epoch, model, criterion, device, use_cuda, mode="val", optimizer=optimizer, writer=None)
            fpr_val, fpr_val_each = get_fpr(output_val, label_val, label2person)
            print("Val fpr:{}".format(fpr_val))
            if acc_val > 0.90:
                save_path = os.path.join(args.ckpt_path, f"LSTM_{epoch}_{args.optim}_lr{args.lr}.pt")
                torch.save(model.state_dict(), save_path)
        writer.flush()
        writer.close()
    
    elif args.mode=="test":
        log_file = os.path.join(args.base_path, "logs", "test_log.log")
        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
        logger.addHandler(fh)

        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.ckpt_name)))
        test_dataset = BreathData_cycle2(data_path, mode="test")
        test_loader = DataLoader(test_dataset, batch_size= args.batch_size, num_workers=8, shuffle=False)
        idx_record = test_dataset.idx_record["test"]
        acc_test, output_test, label_test = run_epoch(test_loader, 0, model, criterion, device, use_cuda, mode="test", 
            optimizer=optimizer, writer=None, idx_record=idx_record)
        fpr_test, fpr_test_each = get_fpr(output_test, label_test, label2person)

        logger.info("Test TPR: {}, Test FPR:{}".format(acc_test, fpr_test))
        logger.info("Test FPR each: {}".format(fpr_test_each))
        logger.info(classification_report(label_test, output_test, target_names=target_names))

        val_dataset = BreathData_cycle2(data_path, mode="val")
        val_loader = DataLoader(val_dataset, batch_size= args.batch_size, num_workers=8, shuffle=False)
        acc_val, output_val, label_val = run_epoch(val_loader, 0, model, criterion, device, use_cuda, mode="val", optimizer=optimizer, writer=None)
        fpr_val, fpr_val_each = get_fpr(output_test, label_test, label2person)
        logger.info("Val TPR:{}, Val FPR:{}".format(acc_val, fpr_val))
        logger.info("Val FPR each: {}".format(fpr_val_each))
        logger.info(classification_report(label_val, output_val, target_names=target_names))

        with open("label_person_mapping_new.pickle","wb") as file:
            pickle.dump(save_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)   
        print(save_mapping)
        print(val_dataset.save_mapping)

# In[ ]:




