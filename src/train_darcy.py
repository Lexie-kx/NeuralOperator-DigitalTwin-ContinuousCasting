import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

from pod_utils import compute_pod_bases
from network import BranchNet, PODDeepONet


def main():
    # ==========================================
    # 0. 自动创建输出文件夹
    # ==========================================
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"  # 🌟 新增模型保存目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==========================================
    # 1. 设置超参数与路径
    # ==========================================
    DATA_PATH = "../data/piececonst_r421_N1024_smooth1.mat"
    ENERGY_THRESHOLD = 0.99
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 300
    SUB_SAMPLE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    # ==========================================
    # 2. 加载与预处理数据
    # ==========================================
    try:
        data = scipy.io.loadmat(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 找不到数据文件，请检查路径: {DATA_PATH}")
        return

    a_data = data['coeff'][:, ::SUB_SAMPLE, ::SUB_SAMPLE]
    u_data = data['sol'][:, ::SUB_SAMPLE, ::SUB_SAMPLE]

    num_samples, spatial_x, spatial_y = a_data.shape
    spatial_res = spatial_x * spatial_y

    a_flat = a_data.reshape(num_samples, spatial_res)
    u_flat = u_data.reshape(num_samples, spatial_res)

    num_train = 800
    a_train, u_train = a_flat[:num_train], u_flat[:num_train]
    a_test, u_test = a_flat[num_train:], u_flat[num_train:]

    # ==========================================
    # 3. 提取空间主模态
    # ==========================================
    phi_np, _, _ = compute_pod_bases(u_train, energy_threshold=ENERGY_THRESHOLD)
    num_modes = phi_np.shape[1]

    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)
    a_train_tensor = torch.tensor(a_train, dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

    train_dataset = TensorDataset(a_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 4. 初始化网络
    # ==========================================
    branch = BranchNet(input_dim=spatial_res, hidden_layers=[512, 256, 128], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    # ==========================================
    # 5. 训练循环
    # ==========================================
    print("\n🚀 开始训练 Darcy 网络 (共 300 轮)...")
    loss_history = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_a, batch_u in train_loader:
            batch_a, batch_u = batch_a.to(device), batch_u.to(device)
            optimizer.zero_grad()
            predictions = model(batch_a)
            loss = criterion(predictions, batch_u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # ==========================================
    # 🌟 6. 新增：自动保存 Darcy 模型权重
    # ==========================================
    model_save_path = os.path.join(MODELS_DIR, "darcy_pod_deeponet.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 【存盘成功】Darcy 模型已保存至: {model_save_path}")

    # ==========================================
    # 7. 可视化与测试
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.yscale('log')
    plt.title("Training Loss Convergence (2D Darcy)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "darcy_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

    model.eval()
    test_idx = 42
    test_input = torch.tensor(a_test[test_idx:test_idx + 1], dtype=torch.float32).to(device)
    true_output = u_test[test_idx]

    with torch.no_grad():
        pred_output = model(test_input).cpu().numpy().squeeze()

    input_2d = a_test[test_idx].reshape(spatial_x, spatial_y)
    true_output_2d = true_output.reshape(spatial_x, spatial_y)
    pred_output_2d = pred_output.reshape(spatial_x, spatial_y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(input_2d, cmap='viridis', origin='lower')
    axes[0].set_title("Input (Permeability)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(true_output_2d, cmap='plasma', origin='lower')
    axes[1].set_title("True Output (Pressure)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_output_2d, cmap='plasma', origin='lower')
    axes[2].set_title("DeepONet Prediction")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "darcy_prediction_result.png"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()