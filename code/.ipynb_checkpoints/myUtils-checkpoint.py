import os
import pickle

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