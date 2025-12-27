"""
数据集类
"""
import torch
from torch.utils.data import Dataset
from utils.data_preprocessor import DomainPreprocessor


class DNSDataset(Dataset):
    """
    DNS 域名数据集
    """
    
    def __init__(self, domains, labels, preprocessor):
        """
        Args:
            domains: 域名列表
            labels: 标签列表 (0=正常, 1=恶意)
            preprocessor: DomainPreprocessor 实例
        """
        self.domains = domains
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.domains)
    
    def __getitem__(self, idx):
        domain = self.domains[idx]
        label = self.labels[idx]
        
        # 转换为 one-hot 编码
        onehot = self.preprocessor.domain_to_onehot(domain)
        
        # 转换为 tensor 并添加通道维度: (1, max_length, vocab_size)
        onehot_tensor = torch.from_numpy(onehot).unsqueeze(0)
        
        return onehot_tensor, torch.tensor(label, dtype=torch.long)

