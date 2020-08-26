#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/4/10 2:54 下午

@author: wanghengzhi
"""
import queue
import math


def get_huffman_tree(xx):
    min_error = 1000000
    split_point = 0
    for i in range(1, xx.shape[0]):
        if abs(xx[:i].sum() - xx[i:].sum()) < min_error:
            min_error = abs(xx[:i].sum() - xx[i:].sum())
            split_point = i
    return split_point


def print_huffman_result(xx, s=0):
    """
    带区间连续限制的huffman编码结果
    """
    q = queue.Queue()
    q.put((1, xx, s))
    while not q.empty():
        n, t, ss = q.get()
        level = math.floor(math.log(n, 2))
        if len(t) <= 1:
            if level < 6:
                print("{},{},{},{},{}".format(n, 0, ss + 1, 0, 0))
        else:
            split_point = get_huffman_tree(t, ss)
            left, right = t[:split_point], t[split_point:]
            q.put((n * 2, left, ss))
            q.put((n * 2 + 1, right, ss + split_point))
            if level < 5:
                print("{},{},{},{},{}".format(n, split_point + 0.5 + ss, 0, n * 2, n * 2 + 1))
            elif level < 6:
                print("{},{},{},{},{}".format(n, 0, split_point + 0.5 + ss, 0, 0))
    return

