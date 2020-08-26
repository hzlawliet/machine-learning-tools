#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/10 2:14 下午

@author: wanghengzhi
"""
import shap
import numpy as np
import pandas as pd


def xgb_feature_importance(tree_booster):
    """
    xgb gain
    """
    feature_importance = tree_booster.get_score(importance_type='gain')
    df_feature_importance_xgb = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['value'])
    df_feature_importance_xgb = df_feature_importance_xgb.reset_index().rename(columns={'index': 'feature'})
    df_feature_importance_xgb.sort_values(by='value', ascending=False, inplace=True)
    return df_feature_importance_xgb


def gbdt_shap_importance(model, raw_data, feature_names):
    """
    树模型类的shap重要性，raw_data一般放测试集
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(raw_data[feature_names])
    fea_importance_df = pd.DataFrame(shap_values)
    fea_importance_df.columns = feature_names
    fea_importance_df = fea_importance_df.apply(lambda x: np.abs(x).sum())
    df_feature_importance_shap = fea_importance_df.reset_index(name='value').rename(columns={'index': 'feature'})
    df_feature_importance_shap.sort_values(by='value', ascending=False, inplace=True)
    return df_feature_importance_shap

