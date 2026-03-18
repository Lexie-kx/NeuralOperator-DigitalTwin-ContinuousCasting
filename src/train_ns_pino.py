import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

from pod_utils import compute_pod_bases
from network import BranchNet, PODDeepONet


# ==========================================
# 🌟 核心升级 1：不仅要二阶导，还需要一阶导数来算对流！
# ==========================================
def precompute_3d_basis_derivatives(phi_np, T, X, Y, dt, dx, dy):
    print("🧮 正在离线计算 3D 时空物理基底的偏导数矩阵 (含一阶与二阶)...")
    num_modes = phi_np.shape[1]
    phi_3d = phi_np.reshape(T, X, Y, num_modes)

    phi_t_3d = np.zeros_like(phi_3d)
    phi_x_3d = np.zeros_like(phi_3d)
    phi_y_3d = np.zeros_like(phi_3d)
    phi_xx_3d = np.zeros_like(phi_3d)
    phi_yy_3d = np.zeros_like(phi_3d)

    win_len_space = min(11, X // 2 * 2 + 1)
    win_len_time = min(11, T // 2 * 2 + 1)

    for i in range(num_modes):
        # 时间导数
        smoothed_t = savgol_filter(phi_3d[:, :, :, i], window_length=win_len_time, polyorder=3, axis=0)
        phi_t_3d[:, :, :, i] = np.gradient(smoothed_t, dt, axis=0)

        # X 方向一阶与二阶
        smooth_x = savgol_filter(phi_3d[:, :, :, i], window_length=win_len_space, polyorder=3, axis=1)
        phi_x = np.gradient(smooth_x, dx, axis=1)
        phi_x_3d[:, :, :, i] = phi_x
        smooth_x2 = savgol_filter(phi_x, window_length=win_len_space, polyorder=3, axis=1)
        phi_xx_3d[:, :, :, i] = np.gradient(smooth_x2, dx, axis=1)

        # Y 方向一阶与二阶
        smooth_y = savgol_filter(phi_3d[:, :, :, i], window_length=win_len_space, polyorder=3, axis=2)
        phi_y = np.gradient(smooth_y, dy, axis=2)
        phi_y_3d[:, :, :, i] = phi_y
        smooth_y2 = savgol_filter(phi_y, window_length=win_len_space, polyorder=3, axis=2)
        phi_yy_3d[:, :, :, i] = np.gradient(smooth_y2, dy, axis=2)

    return (phi_t_3d.reshape(-1, num_modes),
            phi_x_3d.reshape(-1, num_modes), phi_y_3d.reshape(-1, num_modes),
            phi_xx_3d.reshape(-1, num_modes), phi_yy_3d.reshape(-1, num_modes))


# ==========================================
# 🌟 核心升级 2：傅里叶谱方法瞬解速度场 (CFD 级黑科技)
# ==========================================
def get_velocity_from_vorticity(w, dx, dy):
    """
    输入涡度场 w 形状: (Batch, T, X, Y)
    利用 FFT 求解泊松方程，逆推流函数，再求导得到速度场 u 和 v
    """
    Batch, T, X, Y = w.shape
    device = w.device

    # 构建波数矩阵 (Wavenumbers)
    k_x = torch.fft.fftfreq(X, d=dx, device=device) * 2 * np.pi
    k_y = torch.fft.fftfreq(Y, d=dy, device=device) * 2 * np.pi
    kx, ky = torch.meshgrid(k_x, k_y, indexing='ij')

    k_sq = kx ** 2 + ky ** 2
    k_sq[0, 0] = 1.0  # 防止除以零

    # 涡度转换到频域
    w_h = torch.fft.fft2(w)

    # 解泊松方程得到流函数 (频域)
    psi_h = w_h / k_sq
    psi_h[..., 0, 0] = 0.0  # 去除直流分量

    # 在频域求导得到速度
    u_h = 1j * ky * psi_h
    v_h = -1j * kx * psi_h

    # 逆变换回物理空间
    u_vel = torch.fft.ifft2(u_h).real
    v_vel = torch.fft.ifft2(v_h).real

    return u_vel, v_vel


def main():
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    DATA_PATH = r"D:\NavierStokes_V1e-3_N5000_T50\ns_V1e-3_N5000_T50.mat"
    ENERGY_THRESHOLD = 0.95
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 300

    NU = 1e-3
    LAMBDA_PDE = 1e-4  # 对流项极易引发梯度爆炸，初始权重必须极小

    time_steps, spatial_x, spatial_y = 50, 64, 64
    dt = 1.0 / (time_steps - 1)
    dx = 1.0 / spatial_x  # 周期边界，通常域为 [0, 1)
    dy = 1.0 / spatial_y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 计算设备: {device} | 启动满血版 3D NS PI-DeepONet")

    # 1. 安全加载大型数据
    try:
        with h5py.File(DATA_PATH, 'r') as f:
            a_raw = np.array(f['a'][:1000]).transpose()
            u_raw = np.array(f['u'][:1000]).transpose()
    except Exception as e:
        print(f"❌ 数据读取失败: {e}")
        return

    num_samples = a_raw.shape[0]
    input_dim = spatial_x * spatial_y
    output_dim = time_steps * spatial_x * spatial_y

    a_flat = a_raw.reshape(num_samples, input_dim)
    u_flat = u_raw.reshape(num_samples, output_dim)

    num_train = 800
    a_train, u_train = a_flat[:num_train], u_flat[:num_train]

    # 2. 提取基底与计算导数
    phi_np, _, _ = compute_pod_bases(u_train, energy_threshold=ENERGY_THRESHOLD)
    num_modes = phi_np.shape[1]

    # 解包新增的一阶导数
    phi_t_np, phi_x_np, phi_y_np, phi_xx_np, phi_yy_np = precompute_3d_basis_derivatives(
        phi_np, time_steps, spatial_x, spatial_y, dt, dx, dy
    )

    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)
    phi_t_tensor = torch.tensor(phi_t_np, dtype=torch.float32).to(device)
    phi_x_tensor = torch.tensor(phi_x_np, dtype=torch.float32).to(device)
    phi_y_tensor = torch.tensor(phi_y_np, dtype=torch.float32).to(device)
    phi_xx_tensor = torch.tensor(phi_xx_np, dtype=torch.float32).to(device)
    phi_yy_tensor = torch.tensor(phi_yy_np, dtype=torch.float32).to(device)

    a_train_tensor = torch.tensor(a_train, dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

    train_dataset = TensorDataset(a_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    branch = BranchNet(input_dim=input_dim, hidden_layers=[512, 512, 256], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion_data = nn.MSELoss()

    print(f"\n🚀 注入满血版 NS 流体动力学方程... (共 300 轮)")
    loss_history_data = []
    loss_history_pde = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0

        for batch_a, batch_u in train_loader:
            batch_a, batch_u = batch_a.to(device), batch_u.to(device)
            optimizer.zero_grad()

            # 1. 预测所有场
            coeffs = branch(batch_a)
            w_pred = torch.matmul(coeffs, phi_tensor.T)
            w_t = torch.matmul(coeffs, phi_t_tensor.T)
            w_x = torch.matmul(coeffs, phi_x_tensor.T)
            w_y = torch.matmul(coeffs, phi_y_tensor.T)
            w_xx = torch.matmul(coeffs, phi_xx_tensor.T)
            w_yy = torch.matmul(coeffs, phi_yy_tensor.T)

            # 🌟 2. 将 1D 展平的场全部 reshape 回 4D (Batch, Time, X, Y)，准备物理计算
            batch_size = batch_a.shape[0]
            w_4d = w_pred.view(batch_size, time_steps, spatial_x, spatial_y)
            w_x_4d = w_x.view(batch_size, time_steps, spatial_x, spatial_y)
            w_y_4d = w_y.view(batch_size, time_steps, spatial_x, spatial_y)
            w_t_4d = w_t.view(batch_size, time_steps, spatial_x, spatial_y)
            w_xx_4d = w_xx.view(batch_size, time_steps, spatial_x, spatial_y)
            w_yy_4d = w_yy.view(batch_size, time_steps, spatial_x, spatial_y)

            # 🌟 3. FFT 逆推速度场 (极速无损)
            u_vel, v_vel = get_velocity_from_vorticity(w_4d, dx, dy)

            # 🌟 4. 组装终极物理方程 (时间演化 + 非线性对流 = 粘性耗散)
            # ∂ω/∂t + u(∂ω/∂x) + v(∂ω/∂y) - ν(∂²ω/∂x² + ∂²ω/∂y²) = 0
            pde_residual = w_t_4d + (u_vel * w_x_4d) + (v_vel * w_y_4d) - NU * (w_xx_4d + w_yy_4d)

            # 因为是非线性项，极其容易产生个别离群噪点，使用 Huber Loss 保护网络
            loss_pde = torch.nn.functional.smooth_l1_loss(pde_residual, torch.zeros_like(pde_residual))
            loss_data = criterion_data(w_pred, batch_u)

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
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Data: {avg_data_loss:.6f} | PDE (Full NS): {avg_pde_loss:.6f}")

    # 保存与出图
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "ns_full_pino.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_data, label='Data Loss (MSE)', color='blue', linewidth=2)
    plt.plot(loss_history_pde, label='Full NS PDE Loss (Huber)', color='purple', linewidth=2)
    plt.yscale('log')
    plt.title("3D PI-DeepONet: Full Navier-Stokes with Convection")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "ns_full_pino_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()