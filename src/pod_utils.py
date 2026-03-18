import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os


def compute_pod_bases(u_matrix: np.ndarray, energy_threshold: float = 0.99):
    """
    对高维物理场矩阵进行奇异值分解 (SVD)，提取 POD 基向量。
    """
    print("开始计算 POD 模态 (SVD 分解)...")

    # 1. 计算均值并进行去均值化处理
    u_mean = np.mean(u_matrix, axis=0)
    u_centered = u_matrix - u_mean

    # 2. 对矩阵进行奇异值分解
    U_svd, S_svd, Vh_svd = np.linalg.svd(u_centered.T, full_matrices=False)

    # 3. 计算能量占比，决定需要保留多少个模态
    total_energy = np.sum(S_svd ** 2)
    cumulative_energy = np.cumsum(S_svd ** 2) / total_energy

    # 找到刚好大于等于阈值 (例如 99%) 的模态数量
    num_modes = np.searchsorted(cumulative_energy, energy_threshold) + 1

    print("-" * 30)
    print(f"总模态数: {len(S_svd)}")
    print(f"保留 {energy_threshold * 100}% 能量所需的模态数 (Truncation rank): {num_modes}")
    print("-" * 30)

    # 4. 截断基向量 (只保留前 num_modes 列)
    phi_truncated = U_svd[:, :num_modes]

    return phi_truncated, S_svd, cumulative_energy


if __name__ == '__main__':
    # ==========================================
    # 1. 确认数据路径 (请务必核对这里的文件名！)
    # ==========================================
    data_path = "../data/burgers_data_R10.mat"

    if not os.path.exists(data_path):
        print(f"❌ 找不到文件: {data_path}")
        print("请检查 data 文件夹下有没有这个文件，或者文件名是否拼写错误。")
    else:
        try:
            # 2. 加载数据
            data = scipy.io.loadmat(data_path)
            u_data = data['u']  # 取出目标解矩阵

            # 3. 执行 POD 计算 (我们设定保留 99.9% 的能量)
            phi, singular_values, cum_energy = compute_pod_bases(u_data, energy_threshold=0.999)

            # 4. 画出能量累积图
            plt.figure(figsize=(8, 5))
            plt.plot(cum_energy[:50], marker='o', linestyle='-', color='b')
            plt.axhline(y=0.999, color='r', linestyle='--', label='99.9% Energy')
            plt.title("Cumulative Energy of POD Modes (Burgers)")
            plt.xlabel("Number of Modes")
            plt.ylabel("Cumulative Energy")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"❌ 运行出错: {e}")