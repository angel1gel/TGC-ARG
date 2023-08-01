import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split



def processing_data(file):
    
    sequence_codes = []
    # Coding amino acid
    aa_code = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 
               'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 
               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20, 'Y': 21, 'V': 22, 
               'X': 23}
    
    labels = []

    secondary_structure_codes = []
    # Coding secondary structure 
    ss_code = {'C': 1,'E': 2, 'H': 3 }

    with open(file, 'r') as ARf:
        lines = ARf.read().splitlines()    
    
    sequence = []
    for line in lines:
        seq, label, secondary_struct = line.split(",")
        sequence.append(seq)
        labels.append(int(label))
        
        current_seq = []
        for aa in seq:
            current_seq.append(aa_code[aa])
        sequence_codes.append(torch.tensor(current_seq))

        current_ss = []
        for s in secondary_struct:
            current_ss.append(ss_code[s])
        secondary_structure_codes.append(torch.tensor(current_ss))

    data = rnn_utils.pad_sequence(sequence_codes, batch_first=True) 
    data_ss = rnn_utils.pad_sequence(secondary_structure_codes, batch_first=True)

    return data, torch.tensor(labels), data_ss


data, label, data_ss = processing_data("./ARSS-90-seq+stru.csv.csv")

train_data,no_train_data = train_test_split(data, train_size=0.7)
train_label,no_train_label = train_test_split(label, train_size=0.7)
train_ss,no_train_ss = train_test_split(data_ss, train_size=0.7)

test_data,valid_data = train_test_split(no_train_data, train_size=0.5)
test_label,valid_label = train_test_split(no_train_label, train_size=0.5)
test_ss,valid_ss = train_test_split(no_train_ss, train_size=0.5)


train_dataset = Data.TensorDataset(train_data, train_label, train_ss)
test_dataset = Data.TensorDataset(test_data, test_label, test_ss)
valid_dataset = Data.TensorDataset(valid_data, valid_label, valid_ss)

batch_size = 16
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

class TGCARG(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 64
        self.emb_dim = 128

        #sequence feature extraction
        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=1)
        self.gru_seq = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=3, bidirectional=True, dropout=0.2)

        # Reduce the dimension of the embedding
        self.block_seq = nn.Sequential(nn.Linear(144650, 2048),
                                       nn.BatchNorm1d(2048),
                                       nn.LeakyReLU(),
                                       nn.Linear(2048, 1024))

        #SecondStructure feature extraction
        self.embedding_struct = nn.Embedding(4, self.emb_dim, padding_idx=0)
        self.encoder_struct = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_struct= nn.TransformerEncoder(self.encoder_struct, num_layers=1)
        self.gru_struct = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=3, bidirectional=True, dropout=0.2)
        # Reduce the dimension of the embedding
        self.block_struct = nn.Sequential(nn.Linear(144650, 2048),
                                      nn.BatchNorm1d(2048),
                                      nn.LeakyReLU(),
                                      nn.Linear(2048, 1024))
              

        self.block1 = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256))

        self.block2 = nn.Sequential(nn.Linear(256, 8),
                                    nn.ReLU(),
                                    nn.Linear(8, 2))

    def forward(self, x, struct):
        x = self.embedding_seq(x)
        output = self.transformer_encoder_seq(x).permute(1, 0, 2)
        output, hn = self.gru_seq(output)
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        # print(f'hn.type: {hn.type}')
        output = output.reshape(output.shape[0], -1)
        # print(f'output.shape: {output.shape}')
        hn = hn.reshape(output.shape[0], -1)
        # print(f'hn.shape: {output.shape}')
        output = torch.cat([output, hn], 1)
        # print(f'output.shape: {output.shape}')
        output = self.block_seq(output)

        # Process the secondary structure information
        struct = self.embedding_struct(struct)
        struct_output = self.transformer_struct(struct).permute(1, 0, 2)
        struct_output, struct_hn = self.gru_struct(struct_output)
        struct_output = struct_output.permute(1, 0, 2)
        struct_hn = struct_hn.permute(1, 0, 2)
        struct_output = struct_output.reshape(struct_output.shape[0], -1)
        struct_hn = struct_hn.reshape(struct_output.shape[0], -1)
        struct_output = torch.cat([struct_output, struct_hn], 1)
        struct_output = self.block_struct(struct_output)

        # Fusion of features
        representation = torch.cat([output, struct_output], dim=1)
        return self.block1(representation)

    def train_model(self, x, struct):
        with torch.no_grad():
            output = self.forward(x, struct)
        return self.block2(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euc_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euc_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euc_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []

    struct1_ls = []
    struct2_ls = []

    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, struct1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, struct2 = batch[i + int(batch_size / 2)][0], \
                                       batch[i + int(batch_size / 2)][1], \
                                       batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))

        struct1_ls.append(struct1.unsqueeze(0))
        struct2_ls.append(struct2.unsqueeze(0))

    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)

    ss1 = torch.cat(struct1_ls).to(device)
    ss2 = torch.cat(struct2_ls).to(device)

    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2, ss1, ss2


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluate(iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    for x, y, ss in iter:
        x, y, ss = x.to(device), y.to(device), ss.to(device)
        outputs = net.train_model(x, ss)
        outputs_cpu = outputs.cpu()
        y_cpu = y.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data


def caculate(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp, fp, tn, fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # Sensitivity,Recall
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    #F1
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = float(2 * Precision * Recall) / (Precision + Recall)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # Matthew Correlation Coefficient
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    
    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC, Precision, F1, FPR, TPR, precision, recall]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

net = TGCARG().to(device)

lr = 0.00015
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
contrastive_loss = ContrastiveLoss()
model_loss = nn.CrossEntropyLoss(reduction='sum')
best_acc = 0
best_auc = 0
EPOCH = 80
for epoch in range(EPOCH):
    loss_ls = []
    contrast_loss_ls = []
    loss_1_2_ls = []
    t0 = time.time()
    net.train()
    for seq1, seq2, label, label1, label2, ss1, ss2 in train_iter_cont:
        # print(f'seq1.shape: {seq1.shape}; ss1.shape: {ss1.shape}')
        output1 = net(seq1, ss1)
        output2 = net(seq2, ss2)
        output3 = net.train_model(seq1, ss1)
        output4 = net.train_model(seq2, ss2)
        contrast_loss = contrastive_loss(output1, output2, label)
        loss1 = model_loss(output3, label1)
        loss2 = model_loss(output4, label2)
        loss = contrast_loss + loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
        contrast_loss_ls.append(contrast_loss.item())
        loss_1_2_ls.append((loss1 + loss2).item())

    net.eval()
    with torch.no_grad():
        train_performance, train_roc_data, train_prc_data = evaluate(train_iter, net)
        test_performance, test_roc_data, test_prc_data = evaluate(test_iter, net)
        valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_iter, net)

    results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, contrast_loss: {np.mean(contrast_loss_ls):.5f}, loss1_3: {np.mean(loss_1_2_ls):.5f}\n"
    results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
    results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
               + '\n[ACC,\tSE,\tSP,\tAUC,\tMCC,\tPrecision,\tF1]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        test_performance[0], test_performance[1], test_performance[2], test_performance[3],
        test_performance[4],test_performance[5],test_performance[6]) + '\n' + '=' * 60
    results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
               + '\n[ACC,\tSE,\tSP,\tAUC,\tMCC,\tPrecision,\tF1]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
        valid_performance[4],valid_performance[5],valid_performance[6]) + '\n' + '=' * 60
    print(results)
    test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
    test_auc = test_performance[3]
    if test_auc > best_auc:
        best_auc = test_auc
        best_performance = test_performance
        filename = '{}, {}[{:.3f}].pt'.format('TGC-ARG' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
        save_path_pt = os.path.join('./Model', filename)
        # torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
        best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tSE,\tSP,\tAUC,\tMCC,\tPrecision,\tF1]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            best_performance[0], best_performance[1], best_performance[2], best_performance[3],
            best_performance[4],test_performance[5],test_performance[6]) + '\n' + '=' * 60
        best_results += '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tSE,\tSP,\tAUC,\tMCC,\tPrecision,\tF1]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4],valid_performance[5],valid_performance[6]) + '\n' + '=' * 60
        print(best_results)
        best_ROC = test_roc_data
        best_PRC = test_prc_data
        

