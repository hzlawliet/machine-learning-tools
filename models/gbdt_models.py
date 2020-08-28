#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/8/26 5:55 下午

@author: wanghengzhi
"""

import xgboost as xgb
import numpy as np
import lightgbm as lgb
import re


def train_xgb_model(df_train, df_test, fea_list, missing=np.nan):
    Dtrain = xgb.DMatrix(df_train[fea_list], df_train['label'], missing=missing)
    Dtest = xgb.DMatrix(df_test[fea_list], df_test['label'], missing=missing)
    weight = np.sum(Dtrain.get_label()) * 1.0 / (Dtrain.get_label().shape[0] - np.sum(Dtrain.get_label()))
    print('weight:', weight)
    ntrees = 200
    param = {'max_depth': 5, 'eta': 0.1, 'silent': 1,
             'objective': 'binary:logistic', 'eval_metric': 'auc',
             'scale_pos_weight': 1 / weight, 'subsample': 0.8, 'seed': 27,
             'colsample_bytree': 0.8, 'lambda': 0.1, 'missing': missing,
             'base_score': 0.5}
    watchlist = [(Dtrain, 'train'), (Dtest, 'test')]
    bst = xgb.train(param, Dtrain, ntrees, watchlist, early_stopping_rounds=20)
    return bst


def train_lgb_model(x_train, y_train, x_test, y_test, cat_fea):
    D_train = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_fea)
    D_test = lgb.Dataset(x_test, label=y_test, categorical_feature=cat_fea)

    ntrees = 300
    # weight = np.sum(y_train) * 1.0 / (y_train.shape[0] - np.sum(y_train))

    param = {'objective': 'regression_l1', 'learning_rate': 0.1,
             'max_depth': 5, 'colsample_bytree': 0.8, 'early_stopping_round': 20,
             'lambda_l1': 0.1, "metric": 'mae', 'subsample': 0.8}

    # watchlist = [(D_train, 'train'), (D_test, 'test')]
    bst = lgb.train(param, D_train, ntrees, valid_sets=D_test)
    return bst


def save_xgb_model(model_file, out_file):
    filename = model_file
    savename = out_file
    f = open(filename)
    outfile = open(savename, "w+")
    lines = f.readlines()
    f.close()

    tree = 0
    tree_model = []
    node_dict = {}
    feature_map = {}

    max_node = 0

    for line in lines:
        line = line.strip()
        # print line
        if line[:7] == 'booster':
            if line[8] == '0':
                continue
            tree += 1
            for i in range(max_node + 1):
                if i in node_dict:
                    tree_model.append(node_dict[i])
                else:
                    tree_model.append((0, 0, 0, 0, 0, 0))
            outfile.write("%s\t%s\n" % ('0', ";".join([str(xx) for xx in tree_model])))
            tree_model = []
            node_dict = {}
            max_node = 0
            continue
        node_i = re.search('(\d*):', line)
        left_i = re.search('yes=(\d*),', line)
        right_i = re.search('no=(\d*),', line)
        value_i = re.search('leaf=(.*)', line)
        feature_i = re.search('f(\d*)<', line)
        miss_i = re.search('missing=(\d*)', line)
        thre_i = re.search('<(.*)]', line)
        node = line[node_i.start():node_i.end() - 1]
        if left_i:
            left = line[left_i.start() + 4:left_i.end() - 1]
            right = line[right_i.start() + 3:right_i.end() - 1]
            miss = line[miss_i.start() + 8:miss_i.end()]
            feature = line[feature_i.start() + 1:feature_i.end() - 1]
            thre = line[thre_i.start() + 1:thre_i.end() - 1]
            a = feature_map.get(feature, set([]))
            a.add(thre)
            feature_map[feature] = a
            # if thre == '-9.53674e-07':
            #    thre = 0
            value = -1
        else:
            left = -1
            right = -1
            feature = -1
            thre = 0
            value = line[value_i.start() + 5:value_i.end()]
        tree_node = (int(left), int(right), int(miss), int(feature), float(thre), float(value))
        node_dict[int(node)] = tree_node
        if max_node <= int(node):
            max_node = int(node)
    tree += 1
    for i in range(max_node + 1):
        if i in node_dict:
            tree_model.append(node_dict[i])
        else:
            tree_model.append((0, 0, 0, 0, 0, 0))
    outfile.write("%s\t%s\n" % ('0', ";".join([str(xx) for xx in tree_model])))

    outfile.close()
