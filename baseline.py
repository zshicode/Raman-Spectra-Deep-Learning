import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
import argparse
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import KFold
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1200,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='SVM',
                    help='Model')                    

args = parser.parse_args()

set_seed(args.seed,args.use_cuda)

if args.c == 2: v = pd.read_csv('bin.csv').values
else: v = pd.read_csv('multi.csv').values
x = v[:,:args.d]
y = v[:,-1]

def classifier(tid,vid,tag=1):
    if args.c == 2:
        if args.model == 'RF': clf = RandomForestRegressor(max_depth=5,n_estimators=50)
        else: clf = SVR()
    else:
        if args.model == 'RF': clf = RandomForestClassifier(max_depth=5,n_estimators=50)
        else: clf = SVC()
    
    xt = x[tid]
    xv = x[vid]
    yt = y[tid]
    yv = y[vid]
    print('Fold ',tag)
    clf.fit(xt,yt)
    predict = clf.predict(xv)
    if args.c == 2:
        res = binary(yv,predict)
    else:
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