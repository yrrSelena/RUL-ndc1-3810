#!/usr/bin/env python
# coding: utf-8

# # 2-1 燃弧相角分析

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
from mpl_toolkits.mplot3d import Axes3D

import myUtils


# In[2]:


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# In[3]:


root_data_path = '../data/'
feature_path = root_data_path + '2_features/'
src_data_path = root_data_path + '1_processed_data/2_smoothed_data/'
operation_stage = 'arcing'
operation_ftr_path = feature_path + operation_stage + '/'
machine_files = os.listdir(operation_ftr_path)
print(machine_files)


# In[4]:


machine_ftr_dict_file = feature_path + 'machine_ftr_dict.pkl'
if os.path.exists(machine_ftr_dict_file):
    machine_ftr_dict = myUtils.loadPickleFile(machine_ftr_dict_file)
    print('machine_ftr_dict_file is loaded')
else:
    machine_ftr_dict = {}
    for i, tmp_file in enumerate(machine_files):
        machine_id = re.match('(.*).csv', tmp_file).group(1)
        machine_ftr_dict[machine_id] = pd.read_csv(operation_ftr_path + tmp_file)
    myUtils.dumpPickleFile(machine_ftr_dict, machine_ftr_dict_file)
    print('machine_ftr_dict_file is created')


# In[5]:


en_col = ['energy','duration']
cn_col = ['能量','时长']



res_path = '../res/良信项目进展_东南_20201118/'
tmp_res_path = res_path + '相同燃弧相角的波形示意图/'
machine_ids = [str(i) for i in range(1,11)]


# In[7]:

phase_angle_degree_gap = 2
for machine_id in machine_ids:
    machine_ftr_df = machine_ftr_dict[machine_id]
    operation_data_list = myUtils.loadPickleFile(src_data_path + machine_id + '.pkl')
    
    for range_idx, degree_start in enumerate(range(0, 360, phase_angle_degree_gap)):
        degree_end = phase_angle_degree_gap + degree_start
        tmp_df = machine_ftr_df[(machine_ftr_df['A_phase_angle_degree'] >= degree_start) & (machine_ftr_df['A_phase_angle_degree'] <= degree_end)]
        if len(tmp_df) == 0:
            break
        n_hplots = 3
        n_vplots = int(len(tmp_df.index) / n_hplots + 1)
        fig = plt.figure(figsize=(20,5 * n_vplots))
        plt.subplots_adjust(top=0.95,bottom=0,left=0,right=1,hspace=0.5,wspace=0.3)
        title = '燃弧相角为%.1f度时，电压电流波形（'%(tmp_df['A_phase_angle_degree'].tolist()[0])+ machine_id +'号设备）'
        fig.suptitle(title,fontsize=15,y=1)
        print(n_vplots)
        for plot_idx, idx in enumerate(tmp_df.index):
            #print(idx)
            arcing_start = int(tmp_df.loc[idx]['A_arcing_start'])
            arcing_end = int(tmp_df.loc[idx]['A_arcing_end']) + 1
            phase_angle_start = int(tmp_df.loc[idx]['A_phase_angle_start'])
            phase_angle_end = int(tmp_df.loc[idx]['A_phase_angle_end']) + 1
            
            plt.subplot(n_vplots, n_hplots, plot_idx + 1)

            plt.plot(operation_data_list[idx]['UA'][phase_angle_start-200: phase_angle_end])
            plt.plot(operation_data_list[idx]['IA'][phase_angle_start-200: phase_angle_end])
            plt.plot(operation_data_list[idx]['UA'][arcing_start: arcing_end], c = 'r')
            plt.axvline(arcing_start, c='g', ls=':')
            plt.axvline(phase_angle_start, c='gray', ls = '--')
            plt.axvline(phase_angle_end, c='gray', ls = '--')
            plt.axvline(arcing_end - 1, c='g', ls=':')
            plt.title('第' + str(idx) +'次操作，燃弧时长为%.0f'%(tmp_df.loc[idx]['A_arcing_duration']))
            
            plt.xlabel('采样时间')
            plt.ylabel('A相')
        fig.legend(['电压','电流','燃弧时段的电压','燃弧区间','燃弧相角计算区间'], loc = 'upper center', ncol=5, bbox_to_anchor=(0.38, 1.01, 0.3, 0))#,borderaxespad=1)
        fig.savefig(tmp_res_path + title + '.png', dpi=100, bbox_inches ='tight')
        break
    #plt.close(fig)
    break


