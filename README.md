# 🚀 PI-POD-DeepONet: Physics-Informed AI Surrogate for Continuous Casting
**面向连铸结晶器多场耦合的毫秒级神经算子极速求解引擎**

## 📖 项目简介 (Introduction)
传统工艺数字孪生通常依赖商业软件（如 Fluent/ProCAST）进行极其耗时的网格迭代求解，难以满足线上控制的实时性需求。本项目旨在构建基于 **POD-DeepONet (本征正交分解-深度算子网络)** 的 AI 物理替代模型（Surrogate Model）。
通过学习无穷维函数空间的非线性映射，本引擎能够在输入工况参数后，实现“毫秒级”的多物理场瞬间重构，彻底替代传统机理仿真，为连铸工艺的离线快速优化与线上孪生提供底层算法支撑。

## ✨ 核心架构特性 (Key Architecture Features)
1. **无穷维泛化 (Resolution-Invariant):** 彻底摆脱传统 CNN/U-Net 对固定网格分辨率的依赖，算子网络能够在任意连续空间坐标系下进行精准求值。
2. **物理机理嵌入 (Physics-Informed):** 拒绝纯数据驱动的“黑盒”拟合。本架构支持在损失函数中引入流体力学偏微分方程（如 Navier-Stokes 对流扩散项）作为物理软约束 (PDE Loss)。这确保了神经网络重构出的速度场与压力场严格遵循宏观质量与动量守恒定律。
3. **极速降维求解 (Latent Space Mapping):** 通过 POD 提取复杂 3D 物理场的本征模态，将千万级自由度的强耦合偏微分方程求解，转化为低维潜空间中的矩阵映射，实现跨越数量级的加速。

## 🔬 核心算法可行性验证 (Algorithm Validation)
连铸结晶器内部是一个高度复杂的 3D 多场耦合环境，涉及极其非线性的流体对流、以及固液相变界面（糊状区）剧烈的热物性突变。为了确保底层 AI 架构能够在真实的工业高维张量下稳定收敛，本项目采取了**“由浅入深、降维验证”**的严谨科研策略。

在正式引入服务器端的海量连铸仿真数据前，我们首先在本地计算环境下，针对三大极具代表性的基础物理方程进行了递进式测试。这一步不仅彻底打通了 HDF5 巨型工业数据读取与高维空间重塑的代码管道，更全面验证了 POD-DeepONet 架构对“时空演化”、“突变边界”和“三维复杂涡流”的精准映射能力，为后续构建完全体的连铸孪生预测系统奠定了坚实的算法基座。

### 1. 激波与时空演化映射 (1D Burgers Equation)
> 验证 AI 对非线性时空演化与激波锋面（流体突变）的瞬态捕捉能力。
<img width="874" height="242" alt="{C833A85D-4CE9-4FE7-B868-08391D3B2FD0}" src="https://github.com/user-attachments/assets/26e7840f-85ed-4cd2-aeb6-178b6c2b14e6" />


### 2. 复杂突变边界处理 (2D Darcy Flow)
> 验证 AI 对物理属性阶跃边界（模拟结晶器固液相变糊状区）的稳态重构鲁棒性。
<img width="875" height="243" alt="{2888561F-2F00-46C0-AC8D-C8C217CFC3D9}" src="https://github.com/user-attachments/assets/0e42345d-22eb-4593-a382-6945c99f3755" />


### 3. 三维流体力学时空切片 (3D Navier-Stokes)
> 验证 AI 对巨型 3D 复杂流场（模拟结晶器内部三维钢水湍流）的降维与瞬态切片重构能力。
<img width="408" height="345" alt="{4493A856-108C-4BDB-96CA-BA21F7B1C6E8}" src="https://github.com/user-attachments/assets/4b2ac32e-677c-4db3-b284-e867df2d2941" />

## 📂 项目结构 (Repository Structure)
- `src/` : 核心算法源码（包含数据预处理、POD基底提取、网络训练与推理）
- `models/` : 训练完成的算子网络权重文件 (因体积限制未上传)
- `data/` : 1D/2D/3D 物理场训练数据集 (因体积限制未上传)

## 🎯 下一步研发计划 (Roadmap)
- [x] 搭建 POD-DeepONet 底层双分支架构。
- [x] 跑通 1D/2D/3D 基础物理场的离线极速推理。
- [ ] 接入真实连铸结晶器 3D 多场耦合仿真数据。
- [ ] 引入物理约束 (PDE Loss) 与界面加权惩罚 (Weighted Loss)。
