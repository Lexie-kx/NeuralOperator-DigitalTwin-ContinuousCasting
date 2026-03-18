import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os


def load_darcy_data(file_path: str):
    """加载 Darcy Flow 2D 数据集"""
    print(f"📦 正在努力将 1.6GB 的数据加载进内存，请耐心等待几秒到十几秒...\n文件路径: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

    data = scipy.io.loadmat(file_path)
    print("✅ 数据加载成功！")
    return data


def visualize_darcy_2d(data_dict: dict, sample_idx: int = 0):
    """可视化二维 Darcy Flow 的输入场(渗透率)和输出场(压力)"""

    # FNO 官方的 Darcy 数据集，键名通常是 'coeff' (输入) 和 'sol' (输出)
    # 如果你的数据集碰巧是用 'a' 和 'u'，这里也做了兼容
    input_key = 'coeff' if 'coeff' in data_dict else 'a'
    output_key = 'sol' if 'sol' in data_dict else 'u'

    if input_key not in data_dict or output_key not in data_dict:
        print(f"❌ 数据字典中未找到对应的键值。当前包含的键值有: {data_dict.keys()}")
        return

    # 提取张量
    input_field = data_dict[input_key]
    output_field = data_dict[output_key]

    print(f"\n📊 整个数据集的形状:")
    print(f"输入场 (渗透率): {input_field.shape}")
    print(f"输出场 (压力解): {output_field.shape}")

    # 提取单个样本进行画图
    # 形状通常是 (1024, 421, 421)，取出第 sample_idx 个
    input_sample = input_field[sample_idx]
    output_sample = output_field[sample_idx]

    # 开始画图对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图 1：输入场 (通常是分块常数的随机几何图形)
    im1 = axes[0].imshow(input_sample, cmap='viridis', origin='lower')
    axes[0].set_title(f"Input: Permeability Field (Sample {sample_idx})")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 图 2：输出场 (通常是平滑渐变的流场/压力场)
    im2 = axes[1].imshow(output_sample, cmap='plasma', origin='lower')
    axes[1].set_title(f"Output: Pressure/Solution Field (Sample {sample_idx})")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ==========================================
    # 1. 确认数据路径
    # ==========================================
    # 替换为你实际解压出来的文件名
    DARCY_FILE_PATH = "../data/piececonst_r421_N1024_smooth1.mat"

    try:
        # 2. 执行读取
        dataset = load_darcy_data(DARCY_FILE_PATH)

        # 3. 执行可视化 (你可以修改 sample_idx 查看不同的样本)
        visualize_darcy_2d(dataset, sample_idx=0)

    except Exception as e:
        print(f"运行出错: {e}")