"""
快速测试模型是否正常工作（使用模拟数据）
"""
import torch
from model.CNN import DNSCNN

def test_model():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试模型结构")
    print("=" * 60)
    
    # 创建模型
    model = DNSCNN(num_classes=2)
    print(f"模型创建成功")
    
    # 创建模拟输入
    # 根据模型结构，输入应该是 (batch_size, 1, 64, 40)
    batch_size = 4
    sample_input = torch.randn(batch_size, 1, 64, 40)
    print(f"输入形状: {sample_input.shape}")
    
    # 前向传播
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        print(f"输出形状: {output.shape}")
        print(f"✓ 模型前向传播成功！")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")
        return False

if __name__ == "__main__":
    test_model()

