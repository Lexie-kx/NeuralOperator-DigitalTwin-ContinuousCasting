import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

# 导入你写好的网络结构和工具
from network import BranchNet, PODDeepONet
from pod_utils import compute_pod_bases


def plot_spatiotemporal_comparison(true_field, pred_field, error_field, sample_idx, save_dir):
    """
    绘制极具学术感的 Burgers 1D 时空演化对比图
    横轴：空间 (Space)
    纵轴：时间 (Time)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 统一色标，保证对比严谨
    vmin = min(true_field.min(), pred_field.min())
    vmax = max(true_field.max(), pred_field.max())

    # 1. 真实时空场 (Ground Truth)
    im0 = axes[0].imshow(true_field, cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth Spatiotemporal (Sample {sample_idx})", fontsize=14)
    axes[0].set_xlabel("Space (x)")
    axes[0].set_ylabel("Time (t)")
    fig.colorbar(im0, ax=axes[0])

    # 2. 神经算子瞬间重构场 (Prediction)
    im1 = axes[1].imshow(pred_field, cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"POD-DeepONet Instant Mapping", fontsize=14)
    axes[1].set_xlabel("Space (x)")
    axes[1].set_ylabel("Time (t)")
    fig.colorbar(im1, ax=axes[1])

    # 3. 绝对误差 (Absolute Error)
    im2 = axes[2].imshow(error_field, cmap='magma', origin='lower', aspect='auto')
    axes[2].set_title(f"Absolute Error", fontsize=14)
    axes[2].set_xlabel("Space (x)")
    axes[2].set_ylabel("Time (t)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"burgers_spatiotemporal_{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📸 极速重构时空出图成功！已保存至: {save_path}")
    plt.show()


def main():
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"

    # 🔑 请确保这是你电脑上真实的 Burgers .mat 文件路径
    DATA_PATH = r"../data/burgers_data_R10.mat"  # 如果你的路径不一样，请修改这里
    MODEL_WEIGHTS = os.path.join(MODELS_DIR, "burgers_pod_deeponet.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 Burgers 时空极速重构引擎 | 设备: {device}")

    # ==========================================
    # 1. 加载测试集数据
    # ==========================================
    try:
        data = scipy.io.loadmat(DATA_PATH)
        a_data = data['a']
        u_data = data['u']
    except Exception as e:
        print(f"❌ 数据读取失败，请检查路径: {e}")
        return

    num_samples = a_data.shape[0]
    input_dim = a_data.shape[1]  # 根据刚才的报错，这里会自动读取为 8192

    spatial_dim = input_dim
    time_steps = u_data.shape[1] // spatial_dim

    num_train = 800
    a_test = a_data[num_train:num_train + 100]
    u_test = u_data[num_train:num_train + 100]

    # 提取基底
    u_train_full = u_data[:num_train].reshape(num_train, -1)
    phi_np, _, _ = compute_pod_bases(u_train_full, energy_threshold=0.999)

    # 🔑 强制截取 17 个模态！严格对齐仙丹
    num_modes = 17
    phi_np = phi_np[:, :num_modes]
    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)

    # ==========================================
    # 2. 唤醒模型 (严格对齐源码的网络结构)
    # ==========================================
    # 🔑 严格对齐 [256, 256] 两层隐藏层！
    branch = BranchNet(input_dim=input_dim, hidden_layers=[256, 256], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("✅ 模型权重加载成功！结构对齐无误，准备开始时空瞬间推演...")

    # ==========================================
    # 3. 毫秒级时空映射与出图
    # ==========================================
    test_idx = 42  # 挑第 42 个测试样本

    a_input = a_test[test_idx].reshape(1, -1)
    a_tensor = torch.tensor(a_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        start_time = time.time()
        # ⚡️ 瞬间输出完整的时空场！
        u_pred_tensor = model(a_tensor)
        end_time = time.time()
        print(f"⏱️ 瞬间重构完整时空场耗时: {(end_time - start_time) * 1000:.2f} 毫秒")

    # Reshape 成 (时间步, 空间节点) 的 2D 矩阵用于画图
    u_pred = u_pred_tensor.cpu().numpy().reshape(time_steps, spatial_dim)
    u_true = u_test[test_idx].reshape(time_steps, spatial_dim)
    error = np.abs(u_true - u_pred)

    plot_spatiotemporal_comparison(u_true, u_pred, error, sample_idx=test_idx, save_dir=RESULTS_DIR)


if __name__ == '__main__':
    main()