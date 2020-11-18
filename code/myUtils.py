import os
import re
import time
import csv
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def makedirs(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path, 'successfully created.')
    else:
        print(folder_path, 'already exist.')
        
        
def dumpPickleFile(data, file_path):
    pkl_file = open(file_path, 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()

def loadPickleFile(file_path):
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def getMachineIds():
    root_data_path = '../data/'
    raw_data_path = root_data_path + '0_raw_data/'
    machine_folders = os.listdir(raw_data_path)
    # 获得设备编号
    machine_ids = [re.match('(.*)#ToExcel', machine_folder).group(1) for machine_folder in machine_folders]
    return machine_ids