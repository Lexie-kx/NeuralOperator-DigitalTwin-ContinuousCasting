import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入核心模块
from pod_utils import compute_pod_bases
from network import BranchNet, PODDeepONet

def main():
    # ==========================================
    # 0. 自动创建 results 和 models 文件夹
    # ==========================================
    RESULTS_DIR = "../results"
    MODELS_DIR = "../models"  # 🌟 新增：专门用来存放训练好的模型权重
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"📁 结果与模型输出目录已就绪")

    # ==========================================
    # 1. 设置路径与超参数
    # ==========================================
    NS_FILE_PATH = r"D:\NavierStokes_V1e-3_N5000_T50\ns_V1e-3_N5000_T50.mat"

    ENERGY_THRESHOLD = 0.95
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 300
    NUM_TRAIN = 800
    NUM_TEST = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    # ==========================================
    # 2. 智能读取数据 (只取前 1000 个样本防爆内存)
    # ==========================================
    print("\n📦 正在加载 Navier-Stokes 时空数据...")
    try:
        with h5py.File(NS_FILE_PATH, 'r') as f:
            a_data = np.array(f['a'][:1000]).transpose()
            u_data = np.array(f['u'][:1000]).transpose()
    except Exception as e:
        print(f"❌ 读取数据失败，请检查路径。错误: {e}")
        return

    num_samples, spatial_x, spatial_y, time_steps = u_data.shape

    # ==========================================
    # 3. 时空终极展平 (Spatiotemporal Flattening)
    # ==========================================
    input_dim = spatial_x * spatial_y
    output_dim = spatial_x * spatial_y * time_steps

    a_flat = a_data.reshape(num_samples, input_dim)
    u_flat = u_data.reshape(num_samples, output_dim)

    a_train, u_train = a_flat[:NUM_TRAIN], u_flat[:NUM_TRAIN]
    a_test, u_test = a_flat[NUM_TRAIN:NUM_TRAIN + NUM_TEST], u_flat[NUM_TRAIN:NUM_TRAIN + NUM_TEST]

    # ==========================================
    # 4. 提取时空联合模态
    # ==========================================
    print(f"\n🧮 正在提取 {output_dim} 维时空场的主模态 (请耐心等待)...")
    phi_np, _, _ = compute_pod_bases(u_train, energy_threshold=ENERGY_THRESHOLD)
    num_modes = phi_np.shape[1]

    phi_tensor = torch.tensor(phi_np, dtype=torch.float32).to(device)
    a_train_tensor = torch.tensor(a_train, dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

    train_dataset = TensorDataset(a_train_tensor, u_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 5. 初始化终极网络
    # ==========================================
    print(f"\n🧠 正在初始化 POD-DeepONet (截断模态数: {num_modes})...")
    branch = BranchNet(input_dim=input_dim, hidden_layers=[512, 512, 256], num_modes=num_modes)
    model = PODDeepONet(branch_net=branch, pod_bases=phi_tensor).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    # ==========================================
    # 6. 开始训练
    # ==========================================
    print("\n🚀 开始训练时空网络 (共 300 轮)...")
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
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, 当前学习率: {current_lr:.6f}")

    print("✅ 训练完成！")

    # ==========================================
    # 🌟 7. 核心新增：自动保存模型权重 (存盘功能)
    # ==========================================
    model_save_path = os.path.join(MODELS_DIR, "ns_pod_deeponet.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 【重要】模型的宝贵记忆已成功存盘！")
    print(f"路径: {model_save_path}")
    print(f"下次你可以直接加载这个 .pth 文件，无需重新训练即可进行毫秒级预测！\n")

    # ==========================================
    # 8. Loss 可视化与保存
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.yscale('log')
    plt.title("Training Loss Convergence (Navier-Stokes)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    loss_save_path = os.path.join(RESULTS_DIR, "ns_loss_curve.png")
    plt.savefig(loss_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 9. 测试集时空重构验证
    # ==========================================
    print("\n🔍 正在测试集上生成【时空演化】预测图...")
    model.eval()

    test_idx = 42
    test_input = torch.tensor(a_test[test_idx:test_idx + 1], dtype=torch.float32).to(device)
    true_output = u_test[test_idx]

    with torch.no_grad():
        pred_output = model(test_input).cpu().numpy().squeeze()

    true_output_3d = true_output.reshape(spatial_x, spatial_y, time_steps)
    pred_output_3d = pred_output.reshape(spatial_x, spatial_y, time_steps)

    t_steps = [10, 25, 49]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Navier-Stokes Spatiotemporal Prediction\n(Top: True Flow, Bottom: DeepONet Pred)", fontsize=16)

    for i, t in enumerate(t_steps):
        im_true = axes[0, i].imshow(true_output_3d[:, :, t], cmap='RdBu', origin='lower')
        axes[0, i].set_title(f"True - Time Step {t + 1}")
        fig.colorbar(im_true, ax=axes[0, i], fraction=0.046, pad=0.04)
        axes[0, i].axis('off')

        im_pred = axes[1, i].imshow(pred_output_3d[:, :, t], cmap='RdBu', origin='lower')
        axes[1, i].set_title(f"DeepONet - Time Step {t + 1}")
        fig.colorbar(im_pred, ax=axes[1, i], fraction=0.046, pad=0.04)
        axes[1, i].axis('off')

    plt.tight_layout()

    pred_save_path = os.path.join(RESULTS_DIR, "ns_prediction_result.png")
    plt.savefig(pred_save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()