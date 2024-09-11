import torch
import numpy
from scipy.io import loadmat
import os
import random

def loadmat_dir(dir_path):
    '''
    读取.mat文件
    '''
    volumes = [] # 物体的体素表示 (32, 32, 32)
    surfaceSamples = [] # 物体的表面采样点 (3, 1000)
    closestPoints = [] # 标准网格点的最近点，(32, 32, 32, 3)
    
    for file in os.listdir(dir_path):
        if file.endswith('.mat'):
            file_path = os.path.join(dir_path, file)
            data = loadmat(file_path)
            volumes.append(data['Volume'])
            surfaceSamples.append(data['surfaceSamples'])
            closestPoints.append(data['closestPoints'])
    
    volumes = torch.tensor(volumes).to(torch.float32)
    surfaceSamples = torch.tensor(surfaceSamples).permute(0, 2, 1).to(torch.float32)
    closestPoints = torch.tensor(closestPoints).to(torch.float32)

    data = {
        'Volumes': volumes,
        'surfaceSamples': surfaceSamples,
        'closestPoints': closestPoints
    }

    return data

# 生成一个小批量，小批量数据X长度为batch_size
# Volume.shape = (batch_size, 32, 32, 32)
# surfaceSamples.shape = (batch_size, 1000, 3)
# closestPoints = (batch_size, 32, 32, 32, 3)
def data_iter_random(data, batch_size, shuffle=True):
    """使用随机抽样生成一个小批量子序列"""
    # 对data进行shuffle
    num_examples = len(data['Volumes'])
    initial_indices = list(range(0, num_examples))
    if shuffle:
        random.shuffle(initial_indices)
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        Volume = data['Volumes'][initial_indices_per_batch]
        surfaceSamples = data['surfaceSamples'][initial_indices_per_batch]
        closestPoints = data['closestPoints'][initial_indices_per_batch]
        yield Volume, surfaceSamples, closestPoints

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, file_path, batch_size, shuffle=True):
        self.data_iter_fn = data_iter_random
        self.data = loadmat_dir(file_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return self.data_iter_fn(self.data, self.batch_size, self.shuffle)

