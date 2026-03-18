import torch
import torch.nn as nn


class BranchNet(nn.Module):
    """
    分支网络 (Branch Net):
    负责接收初始场，输出 POD 模态的系数。
    """

    def __init__(self, input_dim: int, hidden_layers: list, num_modes: int):
        super(BranchNet, self).__init__()

        # 构建多层感知机 (MLP)
        layers = []
        in_features = input_dim

        # 循环添加隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.GELU())  # 使用 GELU 激活函数，对平滑物理场效果更好
            in_features = hidden_dim

        # 最后一层：输出维度必须等于我们截断保留的 POD 模态数量
        layers.append(nn.Linear(in_features, num_modes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, input_dim)
        """
        return self.network(x)


class PODDeepONet(nn.Module):
    """
    完整的 POD-DeepONet 组装：
    将预测出的系数，与算好的 POD 基向量进行矩阵相乘，重构物理场。
    """

    def __init__(self, branch_net: BranchNet, pod_bases: torch.Tensor):
        super(PODDeepONet, self).__init__()
        self.branch_net = branch_net

        # 将 POD 基向量作为不可训练的参数 (Buffer) 存入模型
        # pod_bases 形状预期为: (spatial_resolution, num_modes)
        self.register_buffer('pod_bases', pod_bases)

    def forward(self, input_field):
        """
        前向传播：预测高维物理场
        """
        # 1. 预测模态系数
        # coefficients 形状: (batch_size, num_modes)
        coefficients = self.branch_net(input_field)

        # 2. 与 POD 基向量做矩阵乘法，重构出完整物理场
        # (batch_size, num_modes) @ (num_modes, spatial_resolution)
        # -> (batch_size, spatial_resolution)
        reconstructed_field = torch.matmul(coefficients, self.pod_bases.T)

        return reconstructed_field


if __name__ == '__main__':
    # ==========================================
    # 模拟测试：验证张量维度是否与你的数据完全匹配
    # ==========================================

    NUM_MODES = 17  # 你刚刚算出来的截断模态数
    SPATIAL_RESOLUTION = 2048  # 原始空间网格点数
    BATCH_SIZE = 32  # 模拟的批次大小

    print("1. 正在初始化 BranchNet...")
    # 输入维度 2048，经过两个 256 维的隐藏层，最后输出 17 个系数
    branch = BranchNet(input_dim=SPATIAL_RESOLUTION,
                       hidden_layers=[256, 256],
                       num_modes=NUM_MODES)

    print("2. 正在模拟加载计算好的 POD 基向量...")
    # 随机生成一个形状为 (2048, 17) 的假 POD 基矩阵用于测试
    dummy_pod_bases = torch.randn(SPATIAL_RESOLUTION, NUM_MODES)

    print("3. 组装 POD-DeepONet...")
    model = PODDeepONet(branch_net=branch, pod_bases=dummy_pod_bases)

    # 4. 模拟输入数据并进行前向传播
    dummy_input = torch.randn(BATCH_SIZE, SPATIAL_RESOLUTION)
    print(f"\n[测试] 输入数据形状: {dummy_input.shape} -> {BATCH_SIZE} 个样本，每个样本 {SPATIAL_RESOLUTION} 个网格点")

    output = model(dummy_input)
    print(f"[测试] 输出数据形状: {output.shape} -> 成功重构出了相同维度的物理场！")

    # 验证输出维度是否等于输入维度
    assert output.shape == dummy_input.shape, "❌ 维度不匹配！"
    print("\n✅ 网络结构测试通过！管道畅通无阻！")