#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pickle import load
import os
from torch.utils.data import DataLoader, TensorDataset
import torch as t
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
import torch.nn as nn
import pandas as pd


def seed_all(seed=2020):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class AdaWeightedLoss(nn.Module):
    def __init__(self, strategy='linear'):
        super(AdaWeightedLoss, self).__init__()
        self.strategy = strategy

    def forward(self, input, target, global_step):
        """
        The reconstruction error will be calculated between x and x', where
        x is a vector of x_dim.

        args:
            input: original values, [bsz,seq,x_dim]
            target: reconstructed values
            global_step: training global step
            strategy: how fast the coefficient w2 shrink to 1.0
        return:
        """
        bsz, seq, x_dim = target.size()

        with t.no_grad():
            # errors: [bsz,seq]
            # w1: [bsz,seq]
            errors = t.sqrt(t.sum((input - target) ** 2, dim=-1))
            error_mean = t.mean(errors, dim=-1)[:, None]
            error_std = t.std(errors, dim=-1)[:, None] + 1e-6
            z_score = (errors - error_mean) / error_std
            neg_z_score = -z_score
            w1 = t.softmax(neg_z_score, dim=-1)

            # exp_z_score: [bsz,seq] -> [bsz,1,seq] -> [bsz,seq,seq] -> [bsz,seq]
            exp_z_score = t.exp(neg_z_score)
            exp_z_score = exp_z_score[:, None, :].repeat(1, seq, 1)
            step_coeff = t.ones(size=(seq, seq), dtype=target.dtype, device=target.device)

            for i in range(seq):
                if self.strategy == 'log':
                    step_coeff[i][i] *= np.log(global_step + np.e - 1)
                elif self.strategy == 'linear':
                    step_coeff[i][i] *= global_step
                elif self.strategy == 'nlog':
                    step_coeff[i][i] *= global_step * np.log(global_step + np.e - 1)
                elif self.strategy == 'quadratic':
                    step_coeff[i][i] *= (global_step ** 2)
                else:
                    raise KeyError('Decay function must be one of [\'log\',\'linear\',\'nlog\',\'quadratic\']')

            exp_z_score = exp_z_score * step_coeff
            w2 = t.sum(exp_z_score, dim=-1) / exp_z_score[:, t.arange(0, seq), t.arange(0, seq)]
            w = w1 * w2
            # normalization
            w = (w / t.sum(w, dim=-1)[:, None])[:, :, None]

        error_matrix = (target - input) ** 2
        return t.sum(error_matrix * w) / (bsz * x_dim)


def normalize(seq):
    return (seq - t.min(seq)) / (t.max(seq) - t.min(seq))


def anomaly_scoring(values, reconstruction_values):
    scores = []
    for v1, v2 in zip(values, reconstruction_values):
        # Ensure both v1 and v2 are PyTorch tensors
        if isinstance(v1, np.ndarray):
            v1 = t.tensor(v1)  # Convert numpy array to PyTorch tensor if needed
        if isinstance(v2, np.ndarray):
            v2 = t.tensor(v2)  # Convert numpy array to PyTorch tensor if needed

        scores.append(t.sqrt(t.sum((v1 - v2) ** 2)))

    return t.Tensor(scores)

def metrics_calculates(values, re_values, labels):


    scores = anomaly_scoring(values, re_values)



    preds, _ = evaluate(labels, scores,adj=False)
    
    preds_,_ = evaluate(labels, scores,adj=True)
    
    print(preds_.shape)



    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    re = recall_score(y_true=labels, y_pred=preds)

    f1_ = f1_score(y_true=labels, y_pred=preds_)
    pre_ = precision_score(y_true=labels, y_pred=preds_)
    re_ = recall_score(y_true=labels, y_pred=preds_)

    auc = roc_auc_score(y_true=labels, y_score=normalize(scores))

    print('F1 score is [%.5f / %.5f] (before adj / after adj), auc score is %.5f.' % (f1, f1_, auc))
    print('Precision score is [%.5f / %.5f], recall score is [%.5f / %.5f].' % (pre, pre_, re, re_))

    return f1_

def metrics_calculate(values, re_values, labels,mind):
    #SMAP mind=[0.1,0.2,0.3,0.4]
    #PSM mind = [0.1,0.2,0.3,0.4]
    #SWAT mind = [0.25,0.25,0.25,0.25]
    #WADI mind = [0.25,0.25,0.25,0.25]

    # mind = [0.1,0.2,0.3,0.4]
    # mind=[0.0,0.0,0.0,1.0]
    if mind is None:
        scores = anomaly_scoring(values, re_values)
    else:
        scores = t.zeros_like(labels[0])
        i = 0
        result = t.chunk(re_values, 4, dim=0)
        for chunk in result:
            score = anomaly_scoring(values, chunk)
            scores = scores + (score*mind[i])
            i = i + 1
        
    preds, _ = evaluate(labels, scores,adj=False)
    preds_,_ = evaluate(labels, scores,adj=True)

    labels = labels.flatten()

    # if preds is None:
    #     print("Error: preds is None.")
    #     preds = t.zeros_like(labels[0])  # 设置一个默认的 Tensor
    # else:
    #     if t.isnan(preds).any():
    #         print("NaN values found in preds.")
    
    # if t.isnan(preds).any() :
    #     print("NaN values found in preds ")
    # # 处理 NaN 值，例如将其替换为 0
    # preds = t.nan_to_num(preds, nan=0.0)
    

    f1 = f1_score(y_true=labels, y_pred=preds,average='binary',pos_label=1,zero_division=1)
    pre = precision_score(y_true=labels, y_pred=preds,average='binary',pos_label=1)
    re = recall_score(y_true=labels, y_pred=preds,average='binary',pos_label=1)
    

    f1_ = f1_score(y_true=labels, y_pred=preds_,average='binary',pos_label=1, zero_division=1)
    pre_ = precision_score(y_true=labels, y_pred=preds_,average='binary',pos_label=1)
    re_ = recall_score(y_true=labels, y_pred=preds_,average='binary',pos_label=1)

    auc = roc_auc_score(y_true=labels, y_score=preds_)

    print('F1 score is [%.5f / %.5f] (before adj / after adj), auc score is %.5f.' % (f1, f1_, auc))
    print('Precision score is [%.5f / %.5f], recall score is [%.5f / %.5f].' % (pre, pre_, re, re_))

    return f1_


def evaluate(labels, scores, step=500, adj=True):
    # best f1

    min_score = min(scores)

    max_score = max(scores)

    best_f1 = 0.0
    best_preds = None
    
    for th in tqdm(t.linspace(min_score, max_score, step), ncols=70):

        preds = (scores > th).numpy().astype(int)

        preds = t.Tensor(preds)


        if adj:
            preds = adjust_predicts(labels, preds)
        preds = t.Tensor(preds)

        f1 = f1_score(y_true=labels, y_pred=preds)


        # if len(set(labels)) > 1 and len(set(preds.numpy())) > 1:
        #     # 计算f1_score
        #     f1 = f1_score(y_true=labels, y_pred=preds)
        # else:
        #     # 如果只有一个类，跳过该步
        #     f1 = 0.0

        if f1 > best_f1:

            best_f1 = f1
            best_preds = preds
    
    return best_preds, best_f1


def adjust_predicts(label, pred=None):
    predict = pred.numpy().astype(bool)
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True
    return predict.astype(int)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = load(f)
        return data



def get_from_one(ts, window_size, stride):
    ts_length = ts.shape[0]
    print(ts_length)
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start:start + window_size])
    return np.array(samples)


def get_from_all(samples, window_size, stride):
    num_samples, window_size_sample, feature_size = samples.shape
    ts_length = (num_samples - 1) * stride + window_size_sample  # 计算目标序列长度
    ts_reconstructed = np.zeros((ts_length, feature_size), dtype=np.float64)  # 确保数据类型为浮点数
    ts_counts = np.zeros(ts_length, dtype=np.float64)  # 用于记录每个位置的叠加次数

    for i, sample in enumerate(samples):
        start = i * stride
        # 确保 sample 是 NumPy 数组并且数据类型正确
        sample = np.asarray(sample, dtype=np.float64)
        # 对 ts_reconstructed 的切片也进行类型转换
        ts_slice = ts_reconstructed[start:start + window_size_sample]
        ts_slice = np.asarray(ts_slice, dtype=np.float64)
        # 执行加法操作
        ts_reconstructed[start:start + window_size_sample] = ts_slice + sample
        ts_counts[start:start + window_size_sample] += 1  # 对应位置计数

    # 防止除以 0 并对重叠部分求平均
    ts_counts = np.maximum(ts_counts, 1)
    ts_counts = ts_counts[:, np.newaxis]  # 扩展维度以进行广播
    ts_reconstructed /= ts_counts  # 对重叠部分取平均

    return ts_reconstructed



def remove_all_same(train_x, test_x):
    remove_idx = []
    for col in range(train_x.shape[1]):
       
        if max(train_x[:, col]) == min(train_x[:, col]):
            remove_idx.append(col)
        else:
            train_x[:, col] = normalize(train_x[:, col])

        if max(test_x[:, col]) == min(test_x[:, col]):
            remove_idx.append(col)
        else:
            test_x[:, col] = normalize(test_x[:, col])

    all_idx = set(range(train_x.shape[1]))
    remain_idx = list(all_idx - set(remove_idx))
    return train_x[:, remain_idx], test_x[:, remain_idx]



if __name__ == '__main__':
  print('ss')
