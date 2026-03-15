"""
数据加载模块
"""
import pickle
import os
import numpy as np

class DataContainer:
    pass

class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('data'):
            return DataContainer
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            return DataContainer

def load_raw_data(file_path='android_25c.SG'):
    """
    加载原始数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        adj: 邻接矩阵
        influ: 影响矩阵列表
    """
    file_path = os.path.join("data", file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}。")
    
    print(f"[*] 正在加载数据: {file_path} ...")
    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            data = RobustUnpickler(f).load()

    adj = None
    if hasattr(data, 'adj_matrix'): 
        adj = data.adj_matrix
    elif isinstance(data, dict) and 'adj_matrix' in data: 
        adj = data['adj_matrix']
    elif isinstance(data, (list, tuple)): 
        adj = data[0]
    
    influ = None
    if hasattr(data, 'influ_mat_list'): 
        influ = data.influ_mat_list
    elif isinstance(data, dict) and 'influ_mat_list' in data: 
        influ = data['influ_mat_list']
    elif isinstance(data, (list, tuple)): 
        influ = data[1]

    if hasattr(adj, 'toarray'): 
        adj = adj.toarray()
    if isinstance(influ, list): 
        influ = np.array(influ)
    
    return adj, influ
