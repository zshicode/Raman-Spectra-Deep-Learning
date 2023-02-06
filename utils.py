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