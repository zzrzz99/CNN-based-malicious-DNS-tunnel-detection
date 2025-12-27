"""
DNS 隧道检测模型 Demo
演示模型的完整训练和评估流程
"""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.CNN import DNSCNN
from utils.data_extractor import extract_all_domains
from utils.data_preprocessor import DomainPreprocessor
from utils.dataset import DNSDataset
from model.tran_test import train_one_epoch, evaluate


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(benign_dir, malicious_dir):
    """
    加载数据集
    
    Args:
        benign_dir: 正常数据集目录
        malicious_dir: 恶意数据集目录
        
    Returns:
        tuple: (域名列表, 标签列表)
    """
    print("=" * 60)
    print("开始提取数据...")
    print("=" * 60)
    
    # 提取正常域名
    print("\n提取正常域名...")
    benign_data = extract_all_domains(benign_dir, label=0)
    print(f"提取到 {len(benign_data)} 个正常域名")
    
    # 提取恶意域名
    print("\n提取恶意域名...")
    malicious_data = extract_all_domains(malicious_dir, label=1)
    print(f"提取到 {len(malicious_data)} 个恶意域名")
    
    # 合并数据
    all_data = benign_data + malicious_data
    random.shuffle(all_data)
    
    domains = [item[0] for item in all_data]
    labels = [item[1] for item in all_data]
    
    print(f"\n总共提取到 {len(domains)} 个域名样本")
    print(f"正常样本: {labels.count(0)} 个")
    print(f"恶意样本: {labels.count(1)} 个")
    
    return domains, labels


def create_dataloaders(domains, labels, preprocessor, train_ratio=0.8, batch_size=32):
    """
    创建训练集和测试集的数据加载器
    
    Args:
        domains: 域名列表
        labels: 标签列表
        preprocessor: 预处理器
        train_ratio: 训练集比例
        batch_size: 批次大小
        
    Returns:
        tuple: (训练集 DataLoader, 测试集 DataLoader)
    """
    # 创建数据集
    dataset = DNSDataset(domains, labels, preprocessor)
    
    # 划分训练集和测试集
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据集划分:")
    print(f"训练集: {train_size} 个样本")
    print(f"测试集: {test_size} 个样本")
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
        device: 设备 (cpu/cuda)
        num_epochs: 训练轮数
        lr: 学习率
        
    Returns:
        list: 训练历史 (loss, train_acc, test_acc)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    print("\n" + "=" * 60)
    print("开始训练模型...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  训练准确率: {train_acc*100:.2f}%")
        print(f"  测试准确率: {test_acc*100:.2f}%")
        print()
    
    return history


def detailed_evaluate(model, test_loader, device):
    """
    详细评估模型性能
    
    Args:
        model: 模型实例
        test_loader: 测试集数据加载器
        device: 设备
        
    Returns:
        dict: 评估指标
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def print_evaluation_results(metrics):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy):  {metrics['accuracy']*100:.2f}%")
    print(f"精确率 (Precision): {metrics['precision']*100:.2f}%")
    print(f"召回率 (Recall):    {metrics['recall']*100:.2f}%")
    print(f"F1 分数:            {metrics['f1']*100:.2f}%")
    print("\n混淆矩阵:")
    print("        预测")
    print("      正常  恶意")
    print(f"正常  {metrics['confusion_matrix'][0][0]:4d}  {metrics['confusion_matrix'][0][1]:4d}")
    print(f"恶意  {metrics['confusion_matrix'][1][0]:4d}  {metrics['confusion_matrix'][1][1]:4d}")


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    BENIGN_DIR = "datasets/base_benign_dataset"
    MALICIOUS_DIR = "datasets/base_malicious_dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.8
    
    # 检查数据集目录
    if not os.path.exists(BENIGN_DIR):
        print(f"错误: 找不到正常数据集目录: {BENIGN_DIR}")
        return
    
    if not os.path.exists(MALICIOUS_DIR):
        print(f"错误: 找不到恶意数据集目录: {MALICIOUS_DIR}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    domains, labels = load_data(BENIGN_DIR, MALICIOUS_DIR)
    
    if len(domains) == 0:
        print("错误: 未能提取到任何域名数据")
        print("请确保已安装 scapy 库: pip install scapy")
        return
    
    # 创建预处理器
    # 根据模型结构，全连接层输入是 64 * 16 * 10
    # 这意味着经过两次池化后，特征图大小是 (64, 16, 10)
    # 如果输入是 (batch, 1, L, C)，经过两次池化后是 (batch, 64, L/4, C/4)
    # 所以需要 L/4 = 16, C/4 = 10，即 L = 64, C = 40
    REQUIRED_LENGTH = 64
    REQUIRED_VOCAB_SIZE = 40
    
    # 创建字符集（限制为40个字符）
    char_set = list('abcdefghijklmnopqrstuvwxyz0123456789.-_')[:REQUIRED_VOCAB_SIZE]
    preprocessor = DomainPreprocessor(max_length=REQUIRED_LENGTH, char_set=char_set)
    print(f"\n字符集大小: {preprocessor.vocab_size} (模型要求: {REQUIRED_VOCAB_SIZE})")
    print(f"域名最大长度: {preprocessor.max_length} (模型要求: {REQUIRED_LENGTH})")
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        domains, labels, preprocessor, 
        train_ratio=TRAIN_RATIO, 
        batch_size=BATCH_SIZE
    )
    
    # 创建模型
    # 根据模型结构，输入应该是 (batch_size, 1, max_length, vocab_size)
    # 经过两次池化后，特征图大小变为 (max_length/4, vocab_size/4)
    # 模型期望全连接层输入是 64 * 16 * 10 = 10240
    # 所以需要 max_length/4 = 64, vocab_size/4 = 40
    # 即 max_length = 256, vocab_size = 160
    
    # 但根据实际模型代码，全连接层是 64 * 16 * 10
    # 这意味着经过池化后的特征图是 16 × 10
    # 如果原始输入是 (1, L, C)，经过两次 kernel_size=2 的池化：
    # L/4 = 16, C/4 = 10
    # 所以 L = 64, C = 40
    
    # 但我们的预处理器输出是 (max_length, vocab_size)
    # 需要调整模型或预处理器的参数
    
    # 为了适配模型，我们需要确保输入维度正确
    # 模型期望: 经过池化后得到 16 × 10 的特征图
    # 所以输入应该是 (1, 64, 40)
    
    # 检查输入维度
    sample_input, _ = next(iter(train_loader))
    print(f"\n输入张量形状: {sample_input.shape}")
    print(f"期望形状: (batch_size, 1, {REQUIRED_LENGTH}, {REQUIRED_VOCAB_SIZE})")
    
    # 创建模型
    model = DNSCNN(num_classes=2).to(device)
    
    # 验证模型维度
    try:
        with torch.no_grad():
            test_output = model(sample_input.to(device))
            print(f"模型输出形状: {test_output.shape}")
            print("✓ 模型维度检查通过")
    except Exception as e:
        print(f"✗ 模型维度不匹配: {e}")
        print(f"实际输入形状: {sample_input.shape}")
        print(f"期望输入形状: (batch_size, 1, {REQUIRED_LENGTH}, {REQUIRED_VOCAB_SIZE})")
        return
    
    # 训练模型
    history = train_model(
        model, train_loader, test_loader, 
        device, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )
    
    # 详细评估
    metrics = detailed_evaluate(model, test_loader, device)
    print_evaluation_results(metrics)
    
    # 保存模型
    model_save_path = "model/demo_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型已保存到: {model_save_path}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

