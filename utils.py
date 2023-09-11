import numpy as np
import torch
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score,matthews_corrcoef,roc_auc_score,roc_curve

def sensitivity(y_true,y_pred):
    return recall_score(y_true,y_pred)

def specificity(y_true,y_pred):
    tn,fp = confusion_matrix(y_true,y_pred)[0]
    if tn+fp!=0: return tn/(tn+fp)
    else: return 1

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def binary(yv,predict):
    fpr,tpr,th = roc_curve(yv,predict)
    pred = np.ones(len(predict))
    for i in range(len(predict)):
        if predict[i]<th[np.argmax(tpr-fpr)]: pred[i] = 0.0
    
    confusion = confusion_matrix(yv,pred)
    
    print(confusion)
    res = [
        roc_auc_score(yv,predict),
        sensitivity(yv,pred),
        specificity(yv,pred),
        precision_score(yv,pred),
        accuracy_score(yv,pred),
        f1_score(yv,pred),
        matthews_corrcoef(yv,pred)
    ]
    return res

def scaler(x):
    return (x-x.min())/(x.max()-x.min())

def laplacian(wmat):
    deg = torch.diag(torch.sum(wmat,dim=0))
    degpow = torch.pow(deg,-0.5)
    degpow[torch.isinf(degpow)] = 0.0
    W = torch.mm(torch.mm(degpow,wmat),degpow)
    return torch.eye(W.shape[0])-W

def neighborhood(feat,k,spec_ang=False):
    # compute C
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(feat.shape[1],1))
    if spec_ang: dmat = 1 - featprod/np.sqrt(smat*smat.T) # 1 - spectral angle
    else: dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:,1:k+1]
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return C

def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=5,spec_ang=False)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g