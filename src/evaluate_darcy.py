import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# 导入你写好的网络结构和工具
from network import BranchNet, PODDeepONet
from pod_utils import compute_pod_bases


def plot_comparison(true_field, pred_field, error_field, sample_idx, save_dir):
    """
    绘制极具学术汇报级质感的三联对比图：真实值 vs 预测值 vs 绝对误差
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 统一色标范围，确保对比的科学性
    vmin = min(true_field.min(), pred_field.min())
    vmax = max(true_field.max(), pred_field.max())

    # 1. Ground Truth (真实物理场)
    im0 = axes[0].imshow(true_field, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (Sample {sample_idx})", fontsize=14)
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Prediction (神经算子重构场)
    im1 = axes[1].imshow(pred_field, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"POD-DeepONet Prediction", fontsize=14)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Absolute Error (绝对误差)
    im2 = axes[2].imshow(error_field, cmap='magma', origin='lower')
    axes[2].set_title(f"Absolute Error", fontsize=14)
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"darcy_reconstruction_{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📸 极速重构出图成功！已保存至: {save_path}")
    plt.show()


def main():
    # ==========================================
    # 1. 路径与环境准备
    # ==========================================
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"

    # 确保这是你电脑上真实的 mat 文件路径
    DATA_PATH = r"../data/piececonst_r421_N1024_smooth1.mat"

    # 加载纯数据驱动版的仙丹
    MODEL_WEIGHTS = os.path.join(MODELS_DIR, "darcy_pod_deeponet.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动离线极速重构引擎 | 设备: {device}")

    # ==========================================
    # 2. 加载测试集数据 (精准对齐 85x85 和 25个模态)
    # ==========================================
    try:
        data = scipy.io.loadmat(DATA_PATH)
        # 强行降采样！将 421x421 变成 85x85
        a_data = data['coeff'][:, ::5, ::5]
        u_data = data['sol'][:, ::5, ::5]
    except Exception as e:
        print(f"❌ 数据读取失败，请检查路径: {e}")
        return

    num_samples = a_data.shape[0]
    grid_size = a_data.shape[1]
    print(f"📊 当前网格尺寸: {grid_size}x{grid_size} = {grid_size ** 2}")

    # 取最后 100 个样本作为纯测试集
    num_train = 800
    a_test = a_data[num_train:num_train + 100]
    u_test = u_data[num_train:num_train + 100]

    # 重算基底 (提取前 800 个训练集来算基底，防止数据穿越)
    a_train_full = a_data[:num_train].reshape(num_train, -1)
    u_train_full = u_data[:num_train].reshape(num_train, -1)

    # 阈值拉高，多算一点模态备用
    phi_np, _, _ = compute_pod_bases(u_train_full, energy_threshold=0.9999)

    # 强行截取前 25 个模态！严格对齐仙丹里的 [7225, 25]
    num_modes = 25
    phi_np = phi_np[:, :num_modes]
    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)

    # ==========================================
    # 3. 唤醒模型 (实例化网络并加载 .pth 权重)
    # ==========================================
    input_dim = grid_size * grid_size  # 7225

    # 🔑 终极修复：彻底对齐训练时的网络神经元数量！
    branch = BranchNet(input_dim=input_dim, hidden_layers=[512, 256, 128], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    # 穿上旧衣服！
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("✅ 模型权重加载成功！准备开始极速推理...")

    # ==========================================
    # 4. 毫秒级在线推理与出图
    # ==========================================
    # 随便挑第 15 个测试样本
    test_idx = 15

    # 准备输入张量
    a_input = a_test[test_idx].reshape(1, -1)
    a_tensor = torch.tensor(a_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 记录开始时间
        start_time = time.time()

        # ⚡️ 核心动作：瞬间重构物理场！
        u_pred_tensor = model(a_tensor)

        # 记录结束时间
        end_time = time.time()
        print(f"⏱️ 单个样本物理场重构耗时: {(end_time - start_time) * 1000:.2f} 毫秒")

    # 把张量转换回 Numpy 并 reshape 成 2D 图像准备画图
    u_pred = u_pred_tensor.cpu().numpy().reshape(grid_size, grid_size)
    u_true = u_test[test_idx]
    error = np.abs(u_true - u_pred)

    # 调用画图函数
    plot_comparison(u_true, u_pred, error, sample_idx=test_idx, save_dir=RESULTS_DIR)


if __name__ == '__main__':
    main()