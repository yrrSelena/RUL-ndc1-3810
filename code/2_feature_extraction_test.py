#!/usr/bin/env python
# coding: utf-8

# # 2 特征提取

# In[1]:


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


# ### 2.0 参数设置

# In[2]:


# 常数定义
CONST_PARAMS_DICT = {
    'frequency': 1 / 200, # 频率
    'U_amplitude': 421 / np.sqrt(3) * np.sqrt(2), #424,电压幅值
    'I_amplitude': 228,  # 电流幅值
    'U_eps': 5, # 可视为电压为0的阈值
    'I_eps': 5, # 可视为电流为0的阈值
    'three_phase': ['A','B','C']  # 三相名称
}


# In[3]:


#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ### 2.1 特征生成

# #### 拟合正弦函数

# In[4]:


import warnings
from scipy.optimize import leastsq
def calSinByParams(x, params):
    '''
    根据x和参数输出sin函数值
    Args:
        x: x轴数据
        params: 参数
        
    Returns:
        对应的正弦值
    '''
    A, f, theta_x, theta_y = params 
    return A * np.sin(2 * np.pi * f * x + theta_x) + theta_y

def calFittingResidual(params, x, y_real):
    '''
    计算原始数据与拟合数据残差
    '''
    return y_real - calSinByParams(x, params)

def getOptFittingParams(data):
    '''
    获得拟合正弦曲线的最佳参数
    
    Args:
        data:待拟合的原始数据
        
    Returns:
        params_opt: 对原始数据进行正弦拟合的最佳参数
            params_opt[0]: A 振幅
            params_opt[1]: f 频率
            params_opt[2]: theta_x x轴方向上偏移
            params_opt[3]: theta_y y轴方向上的偏移
    '''

    # 待拟合数据
    X = np.arange(len(data))
    Y = np.array(data)
    
    # 初始化参数
    Y_sorted = sorted(Y)
    n = min(int(len(Y)/100), 10)
    A = (np.mean(Y_sorted[-n:]) - np.mean(Y_sorted[:n])) / 2
    
    
    # params_init:根据数据初始化拟合参数 
    params_init = [A, CONST_PARAMS_DICT['frequency'], 0, 0]
    
    # 最小二乘得到拟合最优解
    with warnings.catch_warnings():
        try:
            params_opt, _ = leastsq(calFittingResidual, params_init, args = (X, Y), maxfev=5000)
        except Warning as e:
            print(e)
            params_opt = params_init
    
    return params_opt


# #### 计算燃弧范围

# In[5]:

def findArcingRange(U_data, I_data):
    """
    计算燃弧区间
    
    Args:
        U_data: 单次操作的电压数据
        I_data: 单次操作的电流数据
    
    Returns:
        arcing_range: 
            arcing_range[0] = arcing_start 燃弧开始位置
            arcing_range[1] = arcing_end   燃弧结束位置
    """
    
    # 0. 初始化燃弧范围
    arcing_range = [-1, -1]
    
    n_points = 10
    data_len = len(U_data)
    
    # 1.确定燃弧结束位置：电流为0时表示燃弧结束（结束点后n_points对应的电流均小于电流阈值，结束点前n_points对应的电流均大于电流阈值）
    for i in range(data_len - n_points - 1, n_points, -1):
        #判断条件一、在[i - n_points,i]时刻内的电流是否大于0  
        n_Ii_larger_than_0 = 0
        for Ii in I_data[i - n_points : i]:
            if np.abs(Ii) > CONST_PARAMS_DICT['I_eps']:
                n_Ii_larger_than_0 += 1
        
        #判断条件二、在[i + 1, i + n_points]时刻内的电流是否接近0
        n_Ii_equals_0 = 0
        for Ii in I_data[i + 1: i + n_points]:
            if np.abs(Ii) < CONST_PARAMS_DICT['I_eps']:
                n_Ii_equals_0 += 1
        
        #若以上两个条件基本满足，则说明找到燃弧结束位置
        if n_Ii_larger_than_0 > n_points - 3 and n_Ii_equals_0 > n_points - 3:
            arcing_range[1] = i
            break
        
        
    # 2.确定燃弧开始位置
    
    # 方式一、根据残差确定燃弧开始位置，稳态时残差较小，而燃弧时残差较大
    # 截取部分合闸稳态数据，拟合得到稳态时的电压正弦波形
    end_loc = arcing_range[1] - int(1 / CONST_PARAMS_DICT['frequency'] * 5)
    start_loc = end_loc - int(1 / CONST_PARAMS_DICT['frequency'] * 3)
    U_part = np.array(U_data[start_loc : end_loc])
    params_U_close = getOptFittingParams(U_part)
    
    # 计算拟合的电压与原始数据的残差
    U_real = np.array(U_data[start_loc : arcing_range[1]])
    U_pred = calSinByParams(np.arange(len(U_real)), params_U_close)
    U_residual = np.abs(U_real - U_pred)
    
    # 判断在[i - n_points : i]时刻内的电压残差是否接近0
    # 从燃弧结束点开始往前遍历，如果当前点i的电压残差接近0，且[i - n_points : i]区间内的电压残差基本小于0，则找到燃弧起始点i
    arcing_range[0] = arcing_range[1]
    for i in range(len(U_residual) - 1, n_points - 1, -1):
        if U_residual[i] < CONST_PARAMS_DICT['U_eps']:
            n_U_residual_equals_0 = 0
            for u_residual in U_residual[i - n_points : i]:
                if u_residual < CONST_PARAMS_DICT['U_eps']:
                    n_U_residual_equals_0 += 1
             
            if n_U_residual_equals_0 > n_points -3:
                arcing_range[0] = start_loc + i
                break
            
            
    ''' 
    # 方式二、根据实际电压是否超过阈值判断
    for i in range(arcing_range[1], n_points, -1):
        if np.abs(U_data[i]) <= np.abs(params_U_close[0]):#CONST_PARAMS_DICT['U_eps']:
            # 判断条件一、在[i-n_point, i]时刻内的电压是否都小于电压阈值
            n_Ui_equals_0 = 0
            for Ui in U_data[i - n_points : i]:
                if np.abs(Ui) <= np.abs(params_U_close[0]):#< CONST_PARAMS_DICT['U_eps']:
                    n_Ui_equals_0 += 1
             
            if n_Ui_equals_0 > n_points / 2:
                arcing_range[0] = i
                break
    '''          
    return arcing_range

def findArcingRange_test(U_data, I_data, machine_id, op_idx):
    """
    计算燃弧区间
    
    Args:
        U_data: 单次操作的电压数据
        I_data: 单次操作的电流数据
    
    Returns:
        arcing_range: 
            arcing_range[0] = arcing_start 燃弧开始位置
            arcing_range[1] = arcing_end   燃弧结束位置
    """
    
    # 0. 初始化燃弧范围
    arcing_range = [-1, -1]
    
    n_points = 10
    data_len = len(U_data)
    
    # 1.确定燃弧结束位置：电流为0时表示燃弧结束（结束点后n_points对应的电流均小于电流阈值，结束点前n_points对应的电流均大于电流阈值）
    for i in range(data_len - n_points - 1, n_points, -1):
        #判断条件一、在[i - n_points,i]时刻内的电流是否大于0  
        n_Ii_larger_than_0 = 0
        for Ii in I_data[i - n_points : i]:
            if np.abs(Ii) > CONST_PARAMS_DICT['I_eps']:
                n_Ii_larger_than_0 += 1
        
        #判断条件二、在[i + 1, i + n_points]时刻内的电流是否接近0
        n_Ii_equals_0 = 0
        for Ii in I_data[i + 1: i + n_points]:
            if np.abs(Ii) < CONST_PARAMS_DICT['I_eps']:
                n_Ii_equals_0 += 1
        
        #若以上两个条件基本满足，则说明找到燃弧结束位置
        if n_Ii_larger_than_0 > n_points - 3 and n_Ii_equals_0 > n_points - 3:
            arcing_range[1] = i
            break
        
        
    # 2.确定燃弧开始位置
    
    # 方式一、根据残差确定燃弧开始位置，稳态时残差较小，而燃弧时残差较大
    # 截取部分合闸稳态数据，拟合得到稳态时的电压正弦波形
    end_loc = arcing_range[1] - int(1 / CONST_PARAMS_DICT['frequency'] * 5)
    start_loc = end_loc - int(1 / CONST_PARAMS_DICT['frequency'] * 3)
    U_part = np.array(U_data[start_loc : end_loc])
    params_U_close = getOptFittingParams(U_part)
    
    # 计算拟合的电压与原始数据的残差
    U_real = np.array(U_data[start_loc : arcing_range[1]])
    U_pred = calSinByParams(np.arange(len(U_real)), params_U_close)
    U_residual = np.abs(U_real - U_pred)
    
    # 判断在[i - n_points : i]时刻内的电压残差是否接近0
    # 从燃弧结束点开始往前遍历，如果当前点i的电压残差接近0，且[i - n_points : i]区间内的电压残差基本小于0，则找到燃弧起始点i
    arcing_range[0] = arcing_range[1]
    for i in range(len(U_residual) - 1, n_points - 1, -1):
        if U_residual[i] < CONST_PARAMS_DICT['U_eps']:
            n_U_residual_equals_0 = 0
            for u_residual in U_residual[i - n_points : i]:
                if u_residual < CONST_PARAMS_DICT['U_eps']:
                    n_U_residual_equals_0 += 1
             
            if n_U_residual_equals_0 > n_points -3:
                arcing_range[0] = start_loc + i
                break
            
    print('方式1 ',arcing_range)
    arcing_range1 = [arcing_range[0], arcing_range[1]]
    
     
    # 方式二、根据实际电压是否超过阈值判断
    U_threshold = 10
    for i in range(arcing_range[1], n_points, -1):
        if np.abs(U_data[i]) <= U_threshold:#CONST_PARAMS_DICT['U_eps']:
            # 判断条件一、在[i-n_point, i]时刻内的电压是否都小于电压阈值
            n_Ui_equals_0 = 0
            for Ui in U_data[i - n_points : i]:
                if np.abs(Ui) <= U_threshold:#< CONST_PARAMS_DICT['U_eps']:
                    n_Ui_equals_0 += 1
             
            if n_Ui_equals_0 > n_points - 3:
                arcing_range[0] = i
                break
    '''  
    plt.figure(figsize=(20,10))
    plt.plot(U_data[show_data_range])
    plt.plot(I_data[show_data_range])
    #print(U_pred)
    plt.plot(range(start_loc, arcing_range[1]),U_pred)
    plt.plot(U_data[data_range],'r')
    plt.axvline(arcing_range1[0], c = 'k', ls = '--', lw = 2)
    plt.axvline(arcing_range[0], c = 'gray', ls = ':')
    plt.axvline(arcing_range[1], c = 'gray', ls = ':')     
    print('方式2 ',arcing_range)
    '''     
    if arcing_range1[0] != arcing_range[0]:
        show_data_range = np.arange(start_loc, arcing_range[1] + 300)
        data_range = np.arange(arcing_range[0], arcing_range[1] + 1) 
        plt.figure(figsize=(20,10))
        plt.plot(U_data[show_data_range])
        plt.plot(I_data[show_data_range])
        #print(U_pred)
        plt.plot(range(start_loc, arcing_range[1]),U_pred)
        plt.plot(U_data[data_range],'r')
        plt.axvline(arcing_range1[0], c = 'pink', ls = '--')
        plt.axvline(arcing_range[0], c = 'gray', ls = ':')
        plt.axvline(arcing_range1[1], c = 'pink', ls = '--')
        plt.axvline(arcing_range[1], c = 'gray', ls = ':')     
        print('方式2 ',arcing_range)
        plt.xlabel('采样时间')
        plt.legend(['电压','电流','拟合合闸稳态电压正弦波','燃弧时段的电压','方式1根据残差截取燃弧区间','方式2根据电压阈值截取燃弧区间'], loc = 'upper center', ncol=3)
        title = '设备' + machine_id + '第' + str(op_idx + 1) + '次操作'
        plt.title('燃弧区间截取（' + title + '）')
        plt.savefig('../data/2_features/2_arcing_range_fig/' + machine_id + '/燃弧区间选取方式对比图/' + title + '.png', dpi=100, bbox_inches ='tight')
        plt.close()
        
    return arcing_range


# ##### unit test

# In[30]:

def findPhaseAngleRange(U_data, arcing_range):
    '''
    燃弧相角计算区间
    '''
    n_points = 10
    data_len = len(U_data)
    
    # 燃弧区间异常判断
    if arcing_range[1]  == -1:
        print('Error: arcing range')
        return arcing_range
    
    # 1) 燃弧相角计算的起始位置：与燃弧的起始范围一致 
    phase_angle_range = [arcing_range[0], arcing_range[1]]
    
    # 2) 确定燃弧相角计算的结束点：电压从零开始上升的位置
    # 如果当前点i的电压接近0，且[i + 1, i + n_points]区间内的电压基本大于0，则找到结束点i
    for i in range(arcing_range[1] , data_len):
        n_Ui_larger_0 = 0
        if U_data[i] <= 0:
            for Ui in U_data[i + 1: i + n_points]:
                if Ui > 0:
                    n_Ui_larger_0 += 1
            
            if n_Ui_larger_0 >= n_points - 1:
                phase_angle_range[1] = i
                break
    return phase_angle_range


# In[31]:
    
def testArcingRange(operation_data_list):
    for op_idx in range(30):
        machine_df = operation_data_list[op_idx]
        U_data = machine_df['UA']
        I_data = machine_df['IA']
        print(op_idx)
        arcing_range = findArcingRange_test(U_data, I_data, machine_id, op_idx)

def testPhaseAngleRange(operation_data_list):
    for op_idx in range(10):
        machine_df = operation_data_list[op_idx]
        U_data = machine_df['UA']
        I_data = machine_df['IA']
        print(op_idx)
        arcing_range = findArcingRange(U_data, I_data)
        phase_angle_range= findPhaseAngleRange(U_data, arcing_range)
        #print(U_data[phase_angle_range[1] - 220 : phase_angle_range[1] - 120] )
        show_data_range = np.arange(arcing_range[0] - 200, phase_angle_range[1] + 300)
        plt.figure(figsize=(20,10))
        plt.plot(U_data[show_data_range])
        plt.plot(I_data[show_data_range])
        plt.axvline(phase_angle_range[0], c = 'g', ls = ':')
        plt.axvline(phase_angle_range[1], c = 'g', ls = ':')
        plt.xlabel('采样时间')
        plt.legend(['电压','电流','燃弧相角计算区间'], loc = 'upper center', ncol=3)
        title = '设备' + machine_id + '第' + str(op_idx + 1) + '次操作'
        plt.title('燃弧相角计算区间截取（' + title + '）')
        plt.savefig('../data/2_features/2_arcing_range_fig/' + machine_id + '/燃弧相角计算区间/' + title + '.png', dpi=100, bbox_inches ='tight')
        plt.close()   
        
src_data_path = '../data/1_processed_data/2_smoothed_data/'
for machine_id in ['4']:
    operation_data_list = myUtils.loadPickleFile(src_data_path + machine_id + '.pkl')
    print(machine_id, len(operation_data_list))
    
    #testArcingRange(operation_data_list)
    
    testPhaseAngleRange(operation_data_list)
