#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/10 2:11 下午

@author: wanghengzhi
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def evaluate_auc_ks(title, y_test, pred_proba):
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    auc = roc_auc_score(y_test, pred_proba)
    fpr, tpr, thre = roc_curve(y_test, pred_proba, pos_label=1)
    res = pd.DataFrame(columns=('AUC', 'KS'))

    temp = [auc, np.max(tpr - fpr)]
    res.loc[title] = temp
    return res


def evaluate_recall_presion_1(model, test_X, test_Y):
    """
    正样本召回
    """
    pos_all = test_Y.sum()

    total = test_Y.count()

    D_test = xgb.DMatrix(test_X, missing=np.nan)
    pred_proba = model.predict(D_test)
    res = pd.DataFrame(columns=('score', 'total', 'count_0', 'count_1', 'precision_1', 'recall_1'))
    index_i = 0
    for i in range(100, -1, -1):

        neg = 0.0
        pos = 0.0
        cnt = 0.0

        for j in range(test_Y.shape[0]):

            if pred_proba[j] >= i * 0.01:
                cnt += 1
                if test_Y.iloc[j] == 1:
                    pos += 1

        if cnt != 0:
            temp = [i,
                    cnt,
                    cnt - pos,
                    pos,
                    pos / cnt,
                    pos / pos_all
                    ]
            res.loc[index_i] = temp
            index_i = index_i + 1

    return res


def evaluate_recall_presion_0(model, test_X, test_Y):
    """
    负样本的召回
    """
    pos_all = len(test_Y) - test_Y.sum()

    total = test_Y.count()

    D_test = xgb.DMatrix(test_X, missing=np.nan)
    pred_proba = model.predict(D_test)
    res = pd.DataFrame(columns=('score', 'total', 'count_0', 'count_1', 'precision_0', 'recall_0'))
    index_i = 0
    for i in range(100, -1, -1):

        neg = 0.0
        pos = 0.0
        cnt = 0.0

        for j in range(test_Y.shape[0]):

            if 1 - pred_proba[j] >= i * 0.01:
                cnt += 1
                if test_Y.iloc[j] == 0:
                    pos += 1

        if cnt != 0:
            temp = [i,
                    cnt,
                    pos,
                    cnt - pos,
                    pos / cnt,
                    pos / pos_all
                    ]
            res.loc[index_i] = temp
            index_i = index_i + 1
    return res


def print_pr_with_threshold(y_score, y_test, start=0, end=101):
    for i in range(start, end):
        thr = i / 100
        y_pred = np.repeat(0, y_score.shape[0])
        y_pred[y_score > thr] = 1
        test_matrix = confusion_matrix(y_test, y_pred)
        ratio = y_pred.sum() / y_pred.shape[0]
        precision = test_matrix[1][1] / (test_matrix[1][1] + test_matrix[0][1])
        recall = test_matrix[1][1] / (test_matrix[1][1] + test_matrix[1][0])
        print(i + 1, ratio, precision, recall)

