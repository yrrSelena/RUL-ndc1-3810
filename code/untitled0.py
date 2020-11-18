# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:27:46 2020

@author: YRR
"""

import re
import os
import time
import csv
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import myUtils

# 常数定义
CONST_PARAMS_DICT = {
    'frequency': 1 / 200, # 频率
    'U_amplitude': 421 / np.sqrt(3) * np.sqrt(2), #424,电压幅值
    'I_amplitude': 228,  # 电流幅值
    'U_eps': 5, # 可视为电压为0的阈值
    'I_eps': 5, # 可视为电流为0的阈值
    'three_phase': ['A','B','C']  # 三相名称
}

src_data_path = '../data/1_processed_data/2_smoothed_data/'
for machine_id in ['4']:
    operation_data_list = myUtils.loadPickleFile(src_data_path + machine_id + '.pkl')
    print(machine_id, len(operation_data_list))
    
    for op_idx in range(len(operation_data_list)):
        machine_df = operation_data_list[op_idx]
        U_data = machine_df['UA']
        I_data = machine_df['IA']
        arcing_range = findArcingRange(U_data, I_data)
        data_range = np.arange(arcing_range[0], arcing_range[1] + 1)
        plt.figure(figsize=(20,10))
        plt.plot(U_data)
        plt.plot(I_data)
        plt.plot(U_data[data_range],'r')
        plt.axvline(arcing_range[0], c = 'g', ls = ':')
        plt.axvline(arcing_range[1], c = 'g', ls = ':')
        break