#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/30 11:10 上午

@author: wanghengzhi
"""
import seaborn as sns
import matplotlib.pyplot as plt


def attention_heat_map(array, epoch):
    """
    :param array: shape(N_samples, N_attentions), value is the softmax result of attention model
    :param epoch:
    :return: None
    """
    sns.heatmap(array, cmap='Reds')
    plt.title('epoch {} attention score'.format(epoch), fontsize=14)
    plt.show()

