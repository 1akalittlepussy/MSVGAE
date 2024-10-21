import torch
from sklearn.metrics import confusion_matrix

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
def sensitivity_specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    #print(con_mat)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    # print("tn:", tn, "tp:", tp, "fn:", fn, "fp:", fp)
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity, specificity
