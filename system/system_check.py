#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/15 3:13 下午

@author: wanghengzhi
"""
import sys
from sys import getsizeof


def show_memory(unit='GB', threshold=1):
    """查看变量占用内存情况

    :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
    :param threshold: 仅显示内存数值大于等于tvhreshold的变量
    """
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    for i in list(globals().keys()):
        memory = eval("getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)


def print_flush():
    sys.stdout.flush()
