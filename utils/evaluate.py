import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, accuracy_score, f1_score, \
    recall_score
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import torch.nn as nn



def evaluate(predictions: np.ndarray, targets: np.ndarray):
    """
    evaluate model performance
    :param predictions: [n_samples, 12, n_nodes, n_features]
    :param targets: np.ndarray, shape [n_samples, 12, n_nodes, n_features]
    :return: a dict [str -> float]
    """
    eps =1e-6
    targets = targets.squeeze()
    predictions = predictions.squeeze()
    assert targets.shape == predictions.shape
    scores ={}
    loss = nn.BCELoss()
    y_true = targets
    y_pred = predictions
    threshold = 0.5
    y_pred_acc = (np.array(y_pred) >= threshold).astype(int)
    scores['MAE']= mae_np(y_pred, y_true)
    scores['RMSE'] = rmse_np(y_pred, y_true)
    # scores['MAPE']= mape_np(y_pred, y_true)
    # scores['R2'] = r2(y_true, y_pred)


    scores['ROC_AUC'] = auc_np(y_pred, y_true)
    # 针对类别不平衡问题，PR AUC是基于精确度和召回率曲线下的面积，特别适用于数据不均衡的情况。
    # scores['Pr_AUC'] = pr_auc(y_pred,y_true)
    # scores['Precision'] = precision(y_true,y_pred_acc)
    # scores['ACC'] = acc(y_true,y_pred_acc)
    # scores['Recall'] = recall(y_true, y_pred_acc)
    # scores['F1'] = F1Score(y_true, y_pred_acc)
    scores['loss'] = bce_loss(y_pred, y_true)

    return scores


def bce_loss(predictions, targets):
    # 确保预测值在 (0, 1) 范围内，避免log(0)错误
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # 计算BCE损失
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    return loss
def auc_np(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)

def rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(mse_np(preds=preds, labels=labels, null_val=null_val))


def mse_np(preds, labels, null_val=np.nan):
    return np.mean((labels - preds) ** 2)


def mae_np(preds, labels, null_val=np.nan):
    return np.mean(np.abs(labels - preds))


# def mape_np(preds, labels, null_val=np.nan):
#     return np.mean(np.abs((labels - preds) / labels)) * 100


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def pr_auc(preds,labels):
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)

def precision(y_true, y_pred):
    return  precision_score(y_true, y_pred)

def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def F1Score(y_true, y_pred):
    return f1_score(y_true, y_pred)