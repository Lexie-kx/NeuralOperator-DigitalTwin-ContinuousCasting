import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

from network import BranchNet, PODDeepONet
from pod_utils import compute_pod_bases


def plot_ns_snapshots(true_field, pred_field, error_field, sample_idx, time_indices, save_dir):
    """
    绘制极具工业汇报感的 3D 时空流体“切片图”
    横向：不同的时间截面 (t1, t2, t3...)
    纵向：真实场 vs 预测场 vs 绝对误差
    """
    num_snapshots = len(time_indices)
    fig, axes = plt.subplots(3, num_snapshots, figsize=(5 * num_snapshots, 12))

    for i, t_idx in enumerate(time_indices):
        gt = true_field[:, :, t_idx]
        pr = pred_field[:, :, t_idx]
        err = error_field[:, :, t_idx]

        vmin = min(gt.min(), pr.min())
        vmax = max(gt.max(), pr.max())

        im0 = axes[0, i].imshow(gt, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Ground Truth (t={t_idx})", fontsize=14)
        axes[0, i].axis('off')
        fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        im1 = axes[1, i].imshow(pr, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"POD-DeepONet (t={t_idx})", fontsize=14)
        axes[1, i].axis('off')
        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        im2 = axes[2, i].imshow(err, cmap='magma', origin='lower')
        axes[2, i].set_title(f"Absolute Error", fontsize=14)
        axes[2, i].axis('off')
        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"ns_snapshots_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📸 3D流体演化切片出图成功！已保存至: {save_path}")
    plt.show()


def main():
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"

    # 真实路径
    DATA_PATH = r"D:\NavierStokes_V1e-3_N5000_T50\ns_V1e-3_N5000_T50.mat"
    MODEL_WEIGHTS = os.path.join(MODELS_DIR, "ns_pod_deeponet.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 3D Navier-Stokes 流体力学重构引擎 | 设备: {device}")

    try:
        with h5py.File(DATA_PATH, 'r') as f:
            a_data = np.array(f['a'], dtype=np.float32)
            u_data = np.array(f['u'], dtype=np.float32)

            if len(a_data.shape) > 1 and a_data.shape[0] < a_data.shape[-1]:
                a_data = a_data.transpose()
                u_data = u_data.transpose()
    except Exception as e:
        print(f"❌ 数据读取失败，请检查路径: {e}")
        return

    num_samples = a_data.shape[0]
    grid_x = u_data.shape[1]
    grid_y = u_data.shape[2]
    time_steps = u_data.shape[3]

    input_dim = grid_x * grid_y
    print(f"📊 数据维度: 样本数 {num_samples}, 空间 {grid_x}x{grid_y}, 时间步 {time_steps}")

    num_train = 800
    a_test = a_data[num_train:num_train + 100]
    u_test = u_data[num_train:num_train + 100]

    u_train_full = u_data[:num_train].reshape(num_train, -1)
    phi_np, _, _ = compute_pod_bases(u_train_full, energy_threshold=0.999)

    # 🔑 终极修复：严格对齐 24 个模态！
    num_modes = 24
    phi_np = phi_np[:, :num_modes]
    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)

    # 隐藏层无需修改，完美吻合
    branch = BranchNet(input_dim=input_dim, hidden_layers=[512, 512, 256], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("✅ 模型权重加载成功！结构对齐无误，准备进行 3D 切片推理...")

    test_idx = 8

    a_input = a_test[test_idx].reshape(1, -1)
    a_tensor = torch.tensor(a_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        start_time = time.time()
        u_pred_tensor = model(a_tensor)
        end_time = time.time()
        print(f"⏱️ 瞬间重构完整 3D 流场耗时: {(end_time - start_time) * 1000:.2f} 毫秒")

    u_pred = u_pred_tensor.cpu().numpy().reshape(grid_x, grid_y, time_steps)
    u_true = u_test[test_idx]
    error = np.abs(u_true - u_pred)

    t1 = time_steps // 5
    t2 = time_steps // 2
    t3 = time_steps - (time_steps // 5)
    snapshots_to_plot = [t1, t2, t3]

    plot_ns_snapshots(u_true, u_pred, error, sample_idx=test_idx, time_indices=snapshots_to_plot, save_dir=RESULTS_DIR)


if __name__ == '__main__':
    main()