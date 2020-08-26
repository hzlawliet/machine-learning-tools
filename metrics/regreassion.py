#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/10 3:12 下午

@author: wanghengzhi
"""


def get_rank_auc(pred, target):
    """
    回归任务下计算auc，复杂度从O(n^2)降到O(nlogn)
    :param pred:
    :param target:
    :return: auc
    """
    def lowbit(x):
        return x & (-x)

    def get_sum(loc, c):
        res = 0
        while loc > 0:
            res += c[loc]
            loc -= lowbit(loc)
        return res

    def update(loc, delta, c):
        while loc < len(c):
            c[loc] += delta
            loc += lowbit(loc)

    a = []
    for i in range(pred.shape[0]):
        a.append([pred[i], target[i]])

    a = sorted(a, key=lambda x: x[1], reverse=True)
    for i in range(len(a)):
        try:
            if a[i][1] == a[i - 1][1]:
                a[i].append(a[i - 1][2])
            else:
                a[i].append(i + 1)
        except:
            a[i].append(i + 1)

    now = 1
    stat = []
    for i in range(1, len(a)):
        if a[i][2] > a[i - 1][2]:
            stat.append(now)
            now = 0
        now += 1
    stat.append(now)

    sum_dic = {}
    now = 0
    for i in range(len(stat)):
        z = len(stat) - i - 1
        now += stat[z]
        sum_dic[z] = now

    total_pairs = 0
    for i in range(len(stat) - 1):
        total_pairs += stat[i] * sum_dic[i + 1]

    a = sorted(a, key=lambda x: x[0], reverse=True)

    #     print(np.max(np.array(a), 0))
    length = len(a) + 1
    #     print(length)
    c = [0 for x in range(length)]
    #     print(len(c))
    correct_pair = 0
    for i in range(len(a)):
        correct_pair += get_sum(a[i][2] - 1, c)
        update(a[i][2], 1, c)

    print(correct_pair)
    sorting_auc = correct_pair * 1.0 / total_pairs
    print("sorting_auc: %s" % sorting_auc)
    return sorting_auc

