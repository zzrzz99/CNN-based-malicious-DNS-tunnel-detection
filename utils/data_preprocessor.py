"""
数据预处理：将域名转换为字符级 one-hot 编码
"""
import numpy as np
import torch


class DomainPreprocessor:
    """
    域名预处理器
    将域名字符串转换为字符级 one-hot 编码矩阵
    """
    
    def __init__(self, max_length=64, char_set=None):
        """
        Args:
            max_length: 域名最大长度（不足则填充，超出则截断）
            char_set: 字符集，如果为 None 则自动构建
        """
        self.max_length = max_length
        
        if char_set is None:
            # 默认字符集：字母、数字、点、连字符、下划线
            self.char_set = list('abcdefghijklmnopqrstuvwxyz0123456789.-_')
        else:
            self.char_set = char_set
        
        # 创建字符到索引的映射
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.vocab_size = len(self.char_set)
    
    def domain_to_onehot(self, domain):
        """
        将域名转换为 one-hot 编码矩阵
        
        Args:
            domain: 域名字符串
            
        Returns:
            numpy.ndarray: shape (max_length, vocab_size) 的 one-hot 矩阵
        """
        domain = domain.lower()  # 转换为小写
        
        # 初始化矩阵
        onehot_matrix = np.zeros((self.max_length, self.vocab_size), dtype=np.float32)
        
        # 填充或截断到 max_length
        domain_chars = list(domain[:self.max_length])
        
        for i, char in enumerate(domain_chars):
            if char in self.char_to_idx:
                char_idx = self.char_to_idx[char]
                onehot_matrix[i, char_idx] = 1.0
        
        return onehot_matrix
    
    def batch_preprocess(self, domains):
        """
        批量预处理域名
        
        Args:
            domains: 域名列表
            
        Returns:
            torch.Tensor: shape (batch_size, 1, max_length, vocab_size)
        """
        batch = []
        for domain in domains:
            onehot = self.domain_to_onehot(domain)
            batch.append(onehot)
        
        # 转换为 tensor 并添加通道维度
        batch_array = np.array(batch)  # (batch_size, max_length, vocab_size)
        batch_tensor = torch.from_numpy(batch_array)
        # 添加通道维度: (batch_size, 1, max_length, vocab_size)
        batch_tensor = batch_tensor.unsqueeze(1)
        
        return batch_tensor

