import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import KFold
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1200,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='CNN',
                    help='Model')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
        )
        self.pool1 = nn.AvgPool1d(4) # d/4 dim
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=1,
            padding=1,
            dilation=2
        )
        self.pool2 = nn.AvgPool1d(4) # d/16 dim
        self.lin = nn.Linear(args.d//16,args.c)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.tanh(self.pool1(x))
        x = self.conv2(x)
        x = F.tanh(self.pool2(x))
        x = x.squeeze()
        x = self.lin(x)
        x = F.sigmoid(x) 
        return x

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
        self.conv1 = nn.LSTM(
            input_size=args.d,
            hidden_size=args.hidden,
        )
        self.conv2 = nn.LSTM(
            input_size=args.hidden,
            hidden_size=args.hidden,
        )
        self.lin = nn.Linear(args.hidden,args.c)
    
    def forward(self,x):
        x = F.tanh(self.conv1(x)[0])
        x = F.tanh(self.conv2(x)[0])
        x = x.squeeze()
        x = F.sigmoid(self.lin(x))
        return x

if args.c == 2: v = pd.read_csv('bin.csv').values
else: v = pd.read_csv('multi.csv').values
x = v[:,:args.d]
y = v[:,-1]

def classifier(tid,vid,tag=1):
    if args.model == 'CNN': clf = CNNNet()
    else: clf = LSTMNet()
    if args.cuda: clf = clf.cuda()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(x[tid]).float().unsqueeze(1)
    xv = torch.from_numpy(x[vid]).float().unsqueeze(1)
    yt = torch.LongTensor(y[tid])
    yv = y[vid]
    if args.cuda:
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    print('Fold ',tag)
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.cross_entropy(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    if args.c == 2:
        predict = mat[:,1].detach().numpy().flatten() 
        res = binary(yv,predict)
    else:
        predict = mat.argmax(dim=-1)
        res = [
            recall_score(yv,predict,average="weighted"),
            precision_score(yv,predict,average="weighted"),
            accuracy_score(yv,predict),
            f1_score(yv,predict,average="weighted"),
        ]
    return res

if args.c == 2:
    df = pd.DataFrame(columns=['AUC', 'Sn', 'Sp', 'Pre', 'Acc', 'F1', 'Mcc'])  
else:
    df = pd.DataFrame(columns=['Rec', 'Pre', 'Acc', 'F1'])

kf = KFold(n_splits=5,shuffle=True,random_state=args.seed)
for i,(tid,vid) in enumerate(kf.split(x)):
    df.loc[i] = classifier(tid,vid,i)

print(df)
print('mean')
print(df.mean())
print('std')
print(df.std())