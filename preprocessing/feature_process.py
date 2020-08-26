#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/30 11:13 上午

@author: wanghengzhi
"""
import gc


def label_encoder(df, fea_name, min_count=None, top_value=None):
    """Fit label encoder
    :param df: pd.DataFrame()
    :param fea_name:
    :param min_count: min_value_counts
    :param top_value:
    """
    if min_count is not None and top_value is not None :
        raise ValueError("min_count or top_value is both None")

    value_counts_df = df[fea_name].value_counts()
    value_counts_df = value_counts_df.sort_values(ascending=False)

    value_counts_map = {}
    # print(value_counts_df)
    if min_count is not None :
        for col in value_counts_df.index:
            if value_counts_df[col] <= min_count:
                value_counts_df.drop(labels = [col], inplace =True)
    else:
        value_counts_df = value_counts_df[:top_value]
    label = 0
    for col in value_counts_df.index:
        value_counts_map[col] = label
        label +=1
    print(fea_name)
    print(value_counts_map)
    print(label)
    df[fea_name] = df[fea_name].apply(lambda x : value_counts_map[x] if x in value_counts_map else label)
    # with open('feature/' + fea_name + '_label_encoder.conf', 'w+') as f:
    #     for k, v in value_counts_map.items():
    #         print(str(k) + '\t' + str(v), file=f)
    #     print('other' + '\t' + str(label), file=f)
    del value_counts_df
    gc.collect()
    print("finish ", fea_name,' value_counts_map .size', len(value_counts_map))

