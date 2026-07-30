# op_matmul_transpose 算子调试与修改记录

在修复和调通 `op_matmul_transpose` (即 $Q K^T$ 形式的转置矩阵乘法) 算子的过程中，我们为了避免修改底层的 `isa.py` 逻辑（特别是保留 `ldma_mov2rmb_transpose` 强依赖 `tile_n1` 作为第一个维度的断言及设计），集中对 `ops/matmul.py` 文件内的分块与搬运调用进行了修正。

以下是主要的错误定位与具体修改内容：

## 1. 补齐缺失的 K1 分块定义

**问题**：运行时报错 `NameError: name 'K1_L0' is not defined`。
**修改**：在进行 L0 级的 shape 计算处，补充了 `K1` 轴向的分块大小：
```python
M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
M1_L0 = ceil_div(M_L0, M0)
N1_L0 = ceil_div(N_L0, N0)
K1_L0 = ceil_div(K_L0, K0) # 新增: 根据 K_L0 计算 K1_L0 对齐块大小
```

## 2. 修正 K 维度的分块循环逻辑

**问题**：修改前，外层对 `l0_k1_start_in_l2` 的遍历使用了错误的步长 `K_L0` 且越界边界计算未以 `K1` 为单位进行对齐，导致下发给指令计算的 size 不匹配。
**修改**：按照 `K1` 的单位推进循环，并且在 `K1` 和实际的 `K` size 之间做了正确的换算：
```python
# 修改前：
# for l0_k1_start_in_l2 in range(0, K_L2, K_L0):
#     k1_size_l0 = min(K_L0, K_L2 - l0_k1_start_in_l2)
#     l0_k_start_in_l2 = l0_k1_start_in_l2 * K0
#     k_size_l0 = min(K0, K_L2 - l0_k_start_in_l2)

# 修改后：
for l0_k1_start_in_l2 in range(0, K1_L2, K1_L0):
    k1_size_l0 = min(K1_L0, K1_L2 - l0_k1_start_in_l2)
    l0_k_start_in_l2 = l0_k1_start_in_l2 * K0
    # 左侧 L0 buffer 使用 k1_size_l0，右侧计算实际所需提取的大小
    k_size_l0 = min(k1_size_l0 * K0, K_L2 - l0_k_start_in_l2)
    ...
```

## 3. 重构 N 维度的分块循环 (N1 对齐)

**问题**：这是解决内存断言报错（`assert ub_n1kn0.shape[0] == tile_n1`）的最关键点。`isa.ldma_mov2rmb_transpose` 是基于 `n1kn0` 的 Tensor 布局设计的，它理应接收以 `N1` 为单位的坐标起点和长度。原代码基于标量 `N`（步长为 `N_L0`）进行遍历，导致切片索引逻辑在输入 `N1` 时不匹配，引发越界或 Assert 错误。
**修改**：将最外层的 N 循环完全改为针对 `N1_L2` 并以步长 `N1_L0` 遍历，再将其还原到原本控制 bias 和输出大小需要的标量 `N` 上。确保提取右矩阵数据时严格按照 `N1` 单位传入参数。

```python
# 修改前：
# for l0_n_start_in_l2 in range(0, N_L2, N_L0):
#     n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
#     l0_n1_start_in_l2 = l0_n_start_in_l2 // N0
#     n1_size_l0 = min(N1_L0, N1_L2 - l0_n1_start_in_l2)

此代码运行成立条件：n起点始终8对齐且步长是8的倍数，而N_L0并不保证这一点

# 修改后：
for l0_n1_start_in_l2 in range(0, N1_L2, N1_L0):
    n1_size_l0 = min(N1_L0, N1_L2 - l0_n1_start_in_l2)
    l0_n_start_in_l2 = l0_n1_start_in_l2 * N0
    n_size_l0 = min(n1_size_l0 * N0, N_L2 - l0_n_start_in_l2)
```

**遍历单元要反推出其他索引时，优先使用不丢信息的变换。`e.g.: n=n1*N0`遍历`n1`推`n`是稳定的；而`n`推`n1`是需要条件的，因为会丢失余数信息，只有在`n mod N0 = 0`时才等价可逆**
**需要保留坐标全部信息，除法需要对齐条件，如果不能表达`n1`块内偏移，优先在块坐标`n1`空间遍历，再用`乘法`映射回元素坐标`n`。**

调用搬运原语时：
```python
# 将外层拿到的 N1 参数传入以对齐 ldma_mov2rmb_transpose 的需要
right_n1k1n0k0_l0 = isa.ldma_mov2rmb_transpose(
    right_n1kn0_l2, 
    N1_L2, 
    l0_n1_start_in_l2, 
    K_L2, 
    l0_k_start_in_l2, 
    n1_size_l0, 
    k_size_l0
)
```

## 结论
经过以上对 L0 层显存切分 (`Tiling`) 的索引计算逻辑进行修整，算子切分在各个维度 (`K1`, `N1`) 严格保持了整块对齐，能够正常复用且通过 `isa.py` 中所有的 Assert 断言，并且连续进行了大规模尺寸随机性的 Python Unittest 均能无误差通过（`average error rate` 收敛在 0.2% 正常精度范围内）。