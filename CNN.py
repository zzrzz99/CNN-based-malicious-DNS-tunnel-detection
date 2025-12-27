import torch
import torch.nn as nn
import torch.nn.functional as F


class DNSCNN(nn.Module):
    """
    基于卷积神经网络的恶意 DNS 隧道检测模型
    输入：字符级 one-hot 编码后的域名矩阵 (1 × L × C)
    输出：正常 / 隧道的分类概率
    """

    def __init__(self, num_classes=2):
        super(DNSCNN, self).__init__()

        # 第一层卷积 + BN + ReLU + 池化
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=2
        )
        self.bn1 = nn.BatchNorm2d(32)

        # 第二层卷积 + BN + ReLU + 池化
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # 全连接层
        self.fc1 = nn.Linear(64 * 16 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, 1, L, C)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # 展平
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
