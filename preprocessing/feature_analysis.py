#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/10 2:08 下午

@author: wanghengzhi
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def feature_missing(data):
    """
    该函数用来实现特征缺失情况的统计
    :param data: 输入的dataframe
    :return: 缺失情况统计
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() * 1.0 / data.isnull().count() * 100).sort_values(ascending=False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent(%)'])


def sample_distribution(df, label, TargetVar):
    distri_df = pd.DataFrame()
    GROUP = []
    TOTAl = []
    group = sorted(list(set(df[label])))
    for i in group:
        df_current = df.loc[df[label] == i, :]
        total_current = len(df_current)

        GROUP.append(i)
        TOTAl.append(total_current)

    distri_df.loc[:, 'group'] = GROUP
    distri_df.loc[:, 'total'] = TOTAl

    for target_label in TargetVar:
        TARGET = []
        TARGET_RATIO = []
        for i in group:
            df_current = df.loc[df[label] == i, :]
            total_current = len(df_current)
            target_current = int(df_current[target_label].sum())
            target_ratio_current = format(np.round(target_current / total_current, 4), '.2%')

            TARGET.append(target_current)
            TARGET_RATIO.append(target_ratio_current)

        distri_df.loc[:, target_label] = TARGET
        distri_df.loc[:, target_label + '_ratio'] = TARGET_RATIO

    return distri_df


def psi_features_table(InputDf, BenchmarkVar, Benchmark, GroupVar, GroupVal, features):
    psi_df = pd.DataFrame()
    for i in GroupVal:
        psi_df_set = pd.DataFrame(columns=[str(i) + '_PSI'])
        for feature in features:
            expected = (InputDf.loc[InputDf[BenchmarkVar].isin(Benchmark), feature]).dropna(axis=0)
            expected.dropna(axis=0)
            actual = (InputDf.loc[InputDf[GroupVar] == i, feature]).dropna(axis=0)
            psi_score = psi(expected, actual)
            psi_df_set.loc[feature] = round(psi_score, 4)

        psi_df = pd.concat([psi_df, psi_df_set], axis=1)
    return psi_df


def psi(expected_array, actual_array, buckets=10, buckettype='bins'):
    '''Calculate the PSI for a single variable

    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into

    Returns:
       psi_value: calculated PSI value
        '''

    def scale_range(input, min, max):
        input += -(np.min(input))
        if max - min != 0:
            input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    def generate_counts(arr, breakpoints):
        '''Generates counts for each bucket by using the bucket values

        Args:
           arr: ndarray of actual values
           breakpoints: list of bucket values

        Returns:
           counts: counts for elements in each bucket, length of breakpoints array minus one
        '''

        def count_in_range(arr, low, high, start):
            '''Counts elements in array between low and high values.
               Includes value if start is true
            '''
            if start:
                return (len(np.where(np.logical_and(arr >= low, arr <= high))[0]))
            return (len(np.where(np.logical_and(arr > low, arr <= high))[0]))

        counts = np.zeros(len(breakpoints) - 1)

        for i in range(1, len(breakpoints)):
            counts[i - 1] = count_in_range(arr, breakpoints[i - 1], breakpoints[i], i == 1)

        return counts

    expected_percents = generate_counts(expected_array, breakpoints) / len(expected_array)
    actual_percents = generate_counts(actual_array, breakpoints) / len(actual_array)

    def sub_psi(e_perc, a_perc):
        '''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.001
        if e_perc == 0:
            e_perc = 0.001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return (value)

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    return psi_value


def plot_hist_distribution(data, xlim=(0, 20), bins=50):
    """
     plot hist gram for some data
    """
    sns.set(rc={'figure.figsize': (20, 5)})
    plt.figure(figsize=(20, 5))
    data.plot(kind='hist', alpha=0.4, xlim=xlim, bins=bins)
    plt.show()


