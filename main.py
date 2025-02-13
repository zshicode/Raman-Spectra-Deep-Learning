import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import KFold
from utils import *
from transformer import TransformerBlock

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
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
parser.add_argument('--model', type=str, default='CLR',
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
        self.conv = nn.Sequential(
            self.conv1,self.pool1,nn.Tanh(),
            self.conv2,self.pool2,nn.Tanh()
        )
        self.lin = nn.Linear(args.d//16,args.c)
    
    def forward(self,x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.lin(x)
        x = torch.sigmoid(x) 
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
        x = torch.tanh(self.conv1(x)[0])
        x = torch.tanh(self.conv2(x)[0])
        x = x.squeeze()
        x = torch.sigmoid(self.lin(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.lstm = nn.LSTM(
            input_size=args.d,
            hidden_size=args.hidden,
        )
        self.t = TransformerBlock(
            hidden=args.hidden, 
            attn_heads=4, 
            feed_forward_hidden=args.hidden, 
            dropout=0.1
        )
        self.lin = nn.Linear(args.hidden,args.c)
    
    def forward(self,x):
        x = torch.tanh(self.lstm(x)[0])
        x = self.t(x,mask=None)
        x = x.squeeze()
        x = torch.sigmoid(self.lin(x))
        return x

class GraphConv(nn.Module):
    # my implementation of GCN
    def __init__(self,in_dim,out_dim,drop=0.5,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        # self.bias = bias
        # if self.bias:
        #     self.b = nn.Parameter(torch.zeros(1, out_dim))
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x

class GNet(nn.Module):
    def __init__(self,in_dim=args.d,out_dim=args.c,hid_dim=args.hidden,bias=False):
        super(GNet,self).__init__()
        self.res1 = GraphConv(in_dim,hid_dim,bias=bias,activation=F.relu)
        self.res2 = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
    
    def forward(self,g,z):
        h = self.res1(g,z)          
        return self.res2(g,h)

class CLR(CNNNet):
    def __init__(self):
        super(CLR,self).__init__()
    
    def forward(self,x):
        n = x.shape[0]
        tau = 0.1
        xd = F.dropout(x)
        x = torch.cat((x,xd))
        x = self.conv(x)
        z = x.squeeze()
        zz = z/torch.norm(z,dim=-1,keepdim=True)
        sim = -F.log_softmax(torch.mm(zz,zz.t())/tau,dim=-1)
        x = self.lin(z)
        x = x.squeeze()
        x = scaler(x)
        x1 = x[:n]
        x2 = x[n:]
        if self.training:
            return x1,x2,sim
        else:
            return x1

if args.c == 2: v = pd.read_csv('bin.csv').values
else: v = pd.read_csv('multi.csv').values
x = v[:,:args.d]
y = v[:,-1]

def classifier(tid,vid,tag=1):
    if args.model == 'CNN': clf = CNNNet()
    elif args.model == 'LSTM': clf = LSTMNet()
    elif args.model == 'Transformer': clf = Transformer()
    else: clf = CLR()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(x[tid]).float().unsqueeze(1)
    xv = torch.from_numpy(x[vid]).float().unsqueeze(1)
    yt = torch.LongTensor(y[tid])
    yv = y[vid]
    if args.model == 'CLR':
        n = len(xt)
        label = torch.zeros((n,n))
        zero = torch.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if yt[i] == yt[j]: label[i,j] = 1.0
        
        label /= label.sum()
        lab = torch.cat((
            torch.cat((zero,label),dim=1),torch.cat((label,zero),dim=1)
        ),dim=0)
        lap = laplacian(lab)
    
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
        if args.model == 'CLR': 
            lab = lab.cuda()
            lap = lap.cuda()
    
    print('Fold ',tag)
    for e in range(args.epochs):
        clf.train()
        if args.model == 'CLR':
            z1,z2,sim = clf(xt)
            z = torch.cat((z1,z2))
            mmd = torch.matmul(torch.matmul(z.t(),lap),z).mean()
            loss = F.cross_entropy(z1,yt) + F.cross_entropy(z2,yt) + torch.mean(sim*lab) # + mmd
        else:
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

def gclassifier(tid,vid,tag=1):
    clf = GNet()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = x[tid]
    xv = x[vid]
    yt = y[tid]
    yv = y[vid]
    st = norm_adj(xt)
    sv = norm_adj(xv)
    xt = torch.from_numpy(xt).float()
    xv = torch.from_numpy(xv).float()
    yt = torch.LongTensor(yt)
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
        st = st.cuda()
        sv = sv.cuda()
    
    print('Fold ',tag)
    for e in range(args.epochs):
        clf.train()
        z = clf(st,xt)
        loss = F.cross_entropy(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(sv,xv)
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
    if args.model == 'GCN':
        df.loc[i] = gclassifier(tid,vid,i)
    else:
        df.loc[i] = classifier(tid,vid,i)

print(df)
print('mean')
print(df.mean())
print('std')
print(df.std())