import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

from pod_utils import compute_pod_bases
from network import BranchNet, PODDeepONet


# ==========================================
# 🌟 2D 物理约束专区 1：离线计算 2D 空间偏导数
# ==========================================
def precompute_2d_basis_derivatives(phi_np, spatial_x, spatial_y, dx, dy):
    """
    针对 2D 网格，离线计算 POD 基底对 x 和 y 的一阶偏导数 (平滑抗噪版)
    """
    print("🧮 正在离线计算 2D 物理基底的偏导数矩阵...")
    num_modes = phi_np.shape[1]

    # 将一维展平的基底重新捏回 2D，方便求导 (spatial_x, spatial_y, num_modes)
    phi_2d = phi_np.reshape(spatial_x, spatial_y, num_modes)

    phi_x_2d = np.zeros_like(phi_2d)
    phi_y_2d = np.zeros_like(phi_2d)

    win_len = min(21, spatial_x // 2 * 2 + 1)

    # 沿着 X 和 Y 方向分别进行平滑和求导
    for i in range(num_modes):
        # Y 方向 (axis=0)
        smoothed_y = savgol_filter(phi_2d[:, :, i], window_length=win_len, polyorder=3, axis=0)
        phi_y_2d[:, :, i] = np.gradient(smoothed_y, dy, axis=0)

        # X 方向 (axis=1)
        smoothed_x = savgol_filter(phi_2d[:, :, i], window_length=win_len, polyorder=3, axis=1)
        phi_x_2d[:, :, i] = np.gradient(smoothed_x, dx, axis=1)

    # 求完导后再展平回 1D，方便网络做矩阵乘法
    phi_x_flat = phi_x_2d.reshape(-1, num_modes)
    phi_y_flat = phi_y_2d.reshape(-1, num_modes)

    return phi_x_flat, phi_y_flat


# ==========================================
# 🌟 2D 物理约束专区 2：在线计算散度 (Divergence)
# ==========================================
def batch_divergence_2d(q_x, q_y, spatial_x, spatial_y, dx, dy, device):
    """
    利用 PyTorch 张量操作，在线计算批量向量场的散度：div(q) = dq_x/dx + dq_y/dy
    """
    batch_size = q_x.shape[0]

    # 恢复 2D 形状: (Batch, X, Y)
    q_x_2d = q_x.view(batch_size, spatial_x, spatial_y)
    q_y_2d = q_y.view(batch_size, spatial_x, spatial_y)

    # 用 torch.gradient 计算空间导数 (内部使用有限差分)
    # 返回的是一个元组，[0] 是对第一维(y)的导数，[1] 是对第二维(x)的导数
    dq_x_dx = torch.gradient(q_x_2d, spacing=dx, dim=2)[0]
    dq_y_dy = torch.gradient(q_y_2d, spacing=dy, dim=1)[0]

    # 散度 = X方向导数 + Y方向导数，最后展平回去
    divergence = (dq_x_dx + dq_y_dy).view(batch_size, -1)
    return divergence


def main():
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    DATA_PATH = "../data/piececonst_r421_N1024_smooth1.mat"
    ENERGY_THRESHOLD = 0.99
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 300
    SUB_SAMPLE = 5
    LAMBDA_PDE = 1e-4  # 物理 Loss 权重

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 计算设备: {device} | 启动 2D Darcy PI-DeepONet")

    # ==========================================
    # 1. 加载与预处理数据
    # ==========================================
    try:
        data = scipy.io.loadmat(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 找不到数据: {DATA_PATH}")
        return

    # 提取渗透率 a 和 压力解 u
    a_data = data['coeff'][:, ::SUB_SAMPLE, ::SUB_SAMPLE]
    u_data = data['sol'][:, ::SUB_SAMPLE, ::SUB_SAMPLE]

    num_samples, spatial_x, spatial_y = a_data.shape
    spatial_res = spatial_x * spatial_y

    # 假设物理域大小为 1x1
    dx = 1.0 / (spatial_x - 1)
    dy = 1.0 / (spatial_y - 1)

    a_flat = a_data.reshape(num_samples, spatial_res)
    u_flat = u_data.reshape(num_samples, spatial_res)

    num_train = 800
    a_train, u_train = a_flat[:num_train], u_flat[:num_train]

    # ==========================================
    # 2. 提取主模态与 2D 基底导数
    # ==========================================
    phi_np, _, _ = compute_pod_bases(u_train, energy_threshold=ENERGY_THRESHOLD)
    num_modes = phi_np.shape[1]

    # 🌟 离线算出 2D 偏导数矩阵
    phi_x_np, phi_y_np = precompute_2d_basis_derivatives(phi_np, spatial_x, spatial_y, dx, dy)

    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)
    phi_x_tensor = torch.tensor(phi_x_np, dtype=torch.float32).to(device)
    phi_y_tensor = torch.tensor(phi_y_np, dtype=torch.float32).to(device)

    a_train_tensor = torch.tensor(a_train, dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

    train_dataset = TensorDataset(a_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 3. 初始化网络
    # ==========================================
    branch = BranchNet(input_dim=spatial_res, hidden_layers=[512, 256, 128], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()

    # ==========================================
    # 4. 带 2D 散度约束的训练循环
    # ==========================================
    print("\n🚀 开始注入 2D 稳态物理灵魂... (共 300 轮)")
    loss_history_data = []
    loss_history_pde = []

    # Darcy 方程的源项通常设定为 f(x) = 1.0
    f_source = 1.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0

        for batch_a, batch_u in train_loader:
            batch_a, batch_u = batch_a.to(device), batch_u.to(device)
            optimizer.zero_grad()

            # 1. 网络推断系数
            coeffs = branch(batch_a)

            # 2. 预测物理场与极速 2D 导数场 (矩阵乘法瞬间完成)
            u_pred = torch.matmul(coeffs, phi_tensor.T)
            u_x_pred = torch.matmul(coeffs, phi_x_tensor.T)
            u_y_pred = torch.matmul(coeffs, phi_y_tensor.T)

            # 🌟 3. 物理残差组装: -div(a * grad(u)) = f
            # 第一步：计算通量通量 q = a * grad(u)
            q_x = batch_a * u_x_pred
            q_y = batch_a * u_y_pred

            # 第二步：计算散度 div(q)
            div_q = batch_divergence_2d(q_x, q_y, spatial_x, spatial_y, dx, dy, device)

            # 第三步：计算 Darcy 残差 PDE = -div(q) - f = 0
            pde_residual = -div_q - f_source

            # 由于渗透率 a 是分块常数，边界处的导数会极大，我们用 Huber Loss (Smooth L1) 代替 MSE 来抵抗极值噪声
            loss_pde = torch.nn.functional.smooth_l1_loss(pde_residual, torch.zeros_like(pde_residual))

            loss_data = criterion_data(u_pred, batch_u)

            # 加权联合惩罚
            total_loss = loss_data + LAMBDA_PDE * loss_pde

            total_loss.backward()
            optimizer.step()

            epoch_data_loss += loss_data.item()
            epoch_pde_loss += loss_pde.item()

        scheduler.step()
        avg_data_loss = epoch_data_loss / len(train_loader)
        avg_pde_loss = epoch_pde_loss / len(train_loader)
        loss_history_data.append(avg_data_loss)
        loss_history_pde.append(avg_pde_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Data Loss: {avg_data_loss:.6f} | PDE Loss: {avg_pde_loss:.6f}")

    # ==========================================
    # 5. 可视化双缝合曲线
    # ==========================================
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "darcy_pino.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_data, label='Data Loss (MSE)', color='blue', linewidth=2)
    plt.plot(loss_history_pde, label='PDE Loss (Smoothed Physics)', color='green', linewidth=2)
    plt.yscale('log')
    plt.title("2D PI-DeepONet: Darcy Flow Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "darcy_pino_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()