import h5py
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os


def load_ns_data(file_path: str):
    """
    智能加载超大体积的 Navier-Stokes 数据集
    """
    print(f"📦 正在挑战超大数据集，这可能需要几十秒...\n路径: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    # NS 数据集通常大于 2GB，被保存为 Matlab v7.3 (HDF5格式)
    try:
        data = scipy.io.loadmat(file_path)
        a_data = data['a']
        u_data = data['u']
        print("✅ 使用 scipy 读取成功！")
    except NotImplementedError:
        print("🔄 检测到 v7.3 超大文件格式，正在切换至 HDF5 引擎读取...")
        import h5py
        with h5py.File(file_path, 'r') as f:
            # HDF5 读取的维度是反的，需要用 transpose() 翻转回 (样本数, X, Y, 时间)
            a_data = np.array(f['a']).transpose()
            u_data = np.array(f['u']).transpose()
            print("✅ 使用 h5py 读取成功！")

    return a_data, u_data


def visualize_fluid_evolution(a_data: np.ndarray, u_data: np.ndarray, sample_idx: int = 0):
    """
    可视化流体涡度场随时间的演化过程
    """
    print(f"\n📊 终极数据集的三维+时间形状:")
    print(f"输入 (初始状态 t=0): {a_data.shape}")
    print(f"输出 (演化状态 t=1~50): {u_data.shape}")

    # 提取第 sample_idx 个样本
    initial_condition = a_data[sample_idx]
    fluid_evolution = u_data[sample_idx]

    # 获取总时间步 (通常是 50)
    total_time_steps = fluid_evolution.shape[-1]

    # 我们挑 4 个关键时间节点来展示演化过程
    t_steps = [0, total_time_steps // 3, 2 * total_time_steps // 3, total_time_steps - 1]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Navier-Stokes Fluid Evolution (Vorticity) - Sample {sample_idx}", fontsize=16)

    for i, t in enumerate(t_steps):
        # 取出对应时间步的 2D 截面
        current_frame = fluid_evolution[:, :, t]

        im = axes[i].imshow(current_frame, cmap='RdBu', origin='lower')
        axes[i].set_title(f"Time Step: {t + 1}")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].axis('off')  # 关掉坐标轴显得更清晰

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ==========================================
    # 1. 填入你的 D 盘数据路径
    # ==========================================
    # ⚠️ 请确保路径和你的实际存放位置一模一样！
    NS_FILE_PATH = r"D:\NavierStokes_V1e-3_N5000_T50\ns_V1e-3_N5000_T50.mat"

    try:
        a_data, u_data = load_ns_data(NS_FILE_PATH)
        visualize_fluid_evolution(a_data, u_data, sample_idx=42)
    except Exception as e:
        print(f"运行出错: {e}")