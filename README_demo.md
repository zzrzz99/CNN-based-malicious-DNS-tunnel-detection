# DNS 隧道检测模型 Demo

这是一个完整的演示脚本，用于证明基于 CNN 的 DNS 隧道检测模型的可行性。

## 功能特性

- ✅ 从 pcap/pcapng 文件中自动提取 DNS 域名
- ✅ 将域名转换为字符级 one-hot 编码
- ✅ 完整的训练和评估流程
- ✅ 详细的性能指标（准确率、精确率、召回率、F1 分数）
- ✅ 混淆矩阵可视化

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- Scapy（用于解析 pcap 文件）
- scikit-learn（用于评估指标）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

直接运行 demo 脚本：

```bash
python demo.py
```

## 项目结构

```
.
├── demo.py                    # 主演示脚本
├── model/
│   ├── CNN.py                # CNN 模型定义
│   └── tran_test.py          # 训练和评估函数
├── utils/
│   ├── data_extractor.py     # 数据提取模块
│   ├── data_preprocessor.py  # 数据预处理模块
│   └── dataset.py            # 数据集类
└── datasets/
    ├── base_benign_dataset/   # 正常 DNS 流量
    └── base_malicious_dataset/ # 恶意 DNS 隧道流量
```

## 模型说明

模型输入：
- 形状：`(batch_size, 1, 64, 40)`
- 64：域名最大长度
- 40：字符集大小（字母、数字、点、连字符、下划线等）

模型输出：
- 二分类：正常 (0) 或 恶意 (1)

## 输出说明

Demo 会输出：
1. 数据提取进度和统计信息
2. 每个训练 epoch 的损失和准确率
3. 最终评估指标：
   - 准确率 (Accuracy)
   - 精确率 (Precision)
   - 召回率 (Recall)
   - F1 分数
   - 混淆矩阵

## 注意事项

- 首次运行需要从 pcap 文件中提取域名，可能需要一些时间
- 如果数据量很大，建议调整 `BATCH_SIZE` 和 `NUM_EPOCHS` 参数
- 训练后的模型会保存到 `model/demo_model.pth`

## 故障排除

如果遇到 `scapy` 相关错误，请确保已正确安装：
```bash
pip install scapy
```

在 Windows 上，可能需要安装额外的依赖：
```bash
pip install scapy[basic]
```

