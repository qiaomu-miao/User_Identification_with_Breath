import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import os, pickle



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


def classify_one_cycle(model, feat, label2person):
    #predict 1 cycle in testing
    feat = torch.tensor(feat).unsqueeze(0).float()
    model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    feat = feat.to(device)
    model = model.to(device)
    start_time = time.time()
    with torch.no_grad():
        output_logits = model(feat)
    output_probs = nn.functional.softmax(output_logits, dim=1)
    output_label = int(np.argmax(output_probs.squeeze().cpu().numpy()))
    output_probs = output_probs.squeeze().tolist()
    print("output_logits:{}".format(output_logits[0][output_label]))
    end_time=time.time()
    print("Model predicting time {}".format(end_time - start_time))
    #print(output_logits)
    #print(output_probs)
    #print(output_label)
    output_person = label2person[output_label]

    return output_probs, output_label, output_person


if __name__=="__main__":
    import scipy.io.wavfile as wav
    from calcmfcc import calcMFCC_delta_delta

    path = "./storage/Qiaomu.wav"
    (rate,sig) = wav.read(path)
    mfcc_feat = calcMFCC_delta_delta(sig,rate)

    model = RNN_model(14)
    model.load_state_dict(torch.load("models/best_model.pt"))
    with open("models/label_person_mapping.pickle", "rb") as file:
        label_person_mapping = pickle.load(file)
    label2person = label_person_mapping["label2person"]
    print(label2person)
    output_prob, output_person = classify_one_cycle(model, mfcc_feat, label2person)
    print("output_person:{}".format(output_person))