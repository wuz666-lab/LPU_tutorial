# LPU Softmax: Complete Educational Explanation

## 目录
1. [Softmax Computation Requirements](#1-softmax-computation-requirements)
2. [LPU Microarchitecture (Python Simulation)](#2-lpu-microarchitecture-python-simulation)
3. [Correct `op_softmax` Walkthrough](#3-correct-op_softmax-walkthrough)
4. [Key Truth Tables](#4-key-truth-tables)
5. [Role of Each Module in Softmax](#5-role-of-each-module-in-softmax)

---

## 1. Softmax Computation Requirements

### 1.1 什么是矩阵 softmax？——逐行独立计算

**输入不是向量，而是矩阵。** 对于一个 $(M, N)$ 矩阵，softmax 对 **每一行独立** 做归一化，输出同样形状的 $(M, N)$ 矩阵。

```
输入矩阵 X (M×N)                   输出矩阵 Y (M×N)
┌─────────────────────┐           ┌──────────────────────────┐
│ a₁₁  a₁₂  a₁₃  a₁₄ │  softmax  │ s₁₁  s₁₂  s₁₃  s₁₄ │ ← 第1行概率和=1
│ a₂₁  a₂₂  a₂₃  a₂₄ │  ──────→  │ s₂₁  s₂₂  s₂₃  s₂₄ │ ← 第2行概率和=1
│ a₃₁  a₃₂  a₃₃  a₃₄ │           │ s₃₁  s₃₂  s₃₃  s₃₄ │ ← 第3行概率和=1
└─────────────────────┘           └──────────────────────────┘
        M=3行, N=4列                    同样的形状 (3×4)
                                      各行之间互不影响！
```

### 1.2 一行数据的 softmax 五步（数值例子）

以第1行 `[2, 1, -1, 0]` (N=4) 为例，逐步推导：

```
Step 1: 求该行的最大值 → reduce max over N
  max = max(2, 1, -1, 0) = 2
  输入: 1行4个值 → 输出: 1个标量  (N→1 的"规约/reduce")

Step 2: 每个元素减去最大值 (数值稳定技巧)
  sub = [2-2, 1-2, -1-2, 0-2] = [0, -1, -3, -2]
  输入: 1个标量max 广播回4个 → 逐元素相减
  输出: 仍是 4 个值

Step 3: 取指数
  exp = [e^0, e^-1, e^-3, e^-2] ≈ [1.0, 0.368, 0.050, 0.135]
  输出: 仍是 4 个值

Step 4: 求指数和 → reduce sum over N
  sum = 1.0 + 0.368 + 0.050 + 0.135 ≈ 1.553
  输入: 4个值 → 输出: 1个标量  (N→1 的"规约/reduce")

Step 5: 每个exp除以sum (归一化)
  output = [1/1.553, 0.368/1.553, 0.050/1.553, 0.135/1.553]
         ≈ [0.644, 0.237, 0.032, 0.087]
  和 = 1.0 ✓  每个值 ∈ [0,1] ✓
```

**扩展到 M 行**：对每一行重复上述5步。3行→3个max→3个sum→每行独立归一化。行与行之间没有任何数据依赖。

### 1.3 Reduce 的核心作用：把一行压缩成一个数

```
  一行有 N 个元素           reduce(N维度)         得到 1 个标量
  ┌───┬───┬───┬───┐                              ┌───┐
  │ 2 │ 1 │-1 │ 0 │  ── max ──→                  │ 2 │  (max)
  └───┴───┴───┴───┘                              └───┘
  
  ┌───┬───┬───┬───┐                              ┌──────┐
  │1.0│.37│.05│.14│  ── sum ──→                  │1.553 │  (sum)
  └───┴───┴───┴───┘                              └──────┘
  
  M 行 → M 个标量 (每行独立reduce，互不干扰)
```

**为什么 reduce + broadcast 配合使用？**

softmax 公式里 `x - max(x)` 中，`max(x)` 是标量，`x` 是向量——维度不匹配。硬件通过以下模式解决：

```
模式: Reduce → Broadcast → Binary

Step A: Reduce(N→1)   一行N个数 → 1个标量 (存入ARB)
       例如: max([2,1,-1,0]) = 2

Step B: Broadcast(1→N)  将标量复制N份
       例如: 2 → [2, 2, 2, 2]

Step C: Binary        两个N维向量 → 逐元素运算
       例如: [2,1,-1,0] - [2,2,2,2] = [0,-1,-3,-2]
```

**关键理解**：
- **Reduce** = 沿指定维度"压扁"，从 2D 变 1D（或 1D 变 0D 标量）
- **Broadcast** = 把"压扁"的结果"撑开"回原始尺寸
- **Binary** = 两个同尺寸张量做逐元素运算
- 三步可在 **单条 ARU 指令内** 完成！

### 1.4 softmax 三指令的全矩阵形状追踪

用 M=3, N=4 的完整例子，追踪每一步数据形状的变化：

```
初始: input 形状 (3, 4)  即 M=3行, N=4列
      ┌──────────────┐
      │ 2   1  -1  0 │  第1行
      │-1   0   3  1 │  第2行
      │ 1   1   1  1 │  第3行
      └──────────────┘

═══════════════════════════════════════════════════
ARU #1: reduce max over N  →  得到每行的最大值
═══════════════════════════════════════════════════
  输入: (M, N) = (3, 4)
  Reduce: 沿N维取max，每行4个数→1个数
  输出: (M,) = (3,)  →  [2, 3, 1]  ← 3行各自的max
  存入 ARB

═══════════════════════════════════════════════════
ARU #2: sub(减去max) → exp → reduce sum over N
═══════════════════════════════════════════════════
  x1 = input (3,4)          x2 = Broadcast([2,3,1])  (3,4)
       ┌──────────┐              ┌──────────┐
       │ 2  1 -1 0│              │ 2  2  2 2│  每行广播各自max
       │-1  0  3 1│              │ 3  3  3 3│
       │ 1  1  1 1│              │ 1  1  1 1│
       └──────────┘              └──────────┘
  
  sub = x1 - x2:
       ┌──────────────┐
       │ 0  -1  -3 -2 │
       │-4  -3   0 -2 │
       │ 0   0   0  0 │
       └──────────────┘
  
  exp:
       ┌───────────────────┐
       │1.0  0.37 0.05 0.14│
       │0.02 0.05 1.0  0.14│
       │1.0  1.0  1.0  1.0 │
       └───────────────────┘
       
  Reduce sum over N → 每行4个数→1个数:
  输出1: (M, N) exp 存入 UB  (3,4) 矩阵
  输出2: (M,)   sum 存入 ARB  [1.55, 1.20, 4.0]  ← 3行各自的sum

═══════════════════════════════════════════════════
ARU #3: div(除以sum)  →  softmax 结果
═══════════════════════════════════════════════════
  x1 = exp from UB (3,4)   x2 = Broadcast([1.55,1.20,4.0])  (3,4)
  
  div = x1 / x2:
       ┌──────────────────────┐
       │0.644  0.237 0.032 0.087│  ← 第1行: 和≈1.0
       │0.015  0.040 0.833 0.113│  ← 第2行: 和≈1.0
       │0.25   0.25  0.25  0.25 │  ← 第3行: 和=1.0
       └──────────────────────┘
  
  输出: (3, 4) softmax结果矩阵
```

### 1.5 硬件约束

| 约束 | 说明 |
|------|------|
| **M 维分块 (Tiling)** | 矩阵太大无法一次装入 L0，在 M 维度切分为多个 tile，逐个处理 |
| **N 维不分块** | reduce 需要看到完整的一行才能正确计算 max/sum，所以 N 方向必须一次性处理 |
| **N0=8 对齐** | N 按 8 对齐分块存储，非 8 倍数的末尾补零（padding），但 reduce 前必须 clip 掉 |
| **单指令多功能融合** | 一条 ARU 指令 = 数据读取 + Broadcast + Binary + Unary + Reduce + 写回（全部可同时配置） |

---

## 2. LPU Microarchitecture (Python Simulation)

### 2.1 整体架构

LPU 是一个模拟的深度学习加速器，通过 Python 类模拟硬件行为。整个仿真架构分为三层：

```
┌─────────────────────────────────────────────────┐
│  算子层 (ops/)                                   │
│  op_softmax, op_matmul, op_layernorm, ...        │
│  描述计算的顶层逻辑与分块策略                       │
├─────────────────────────────────────────────────┤
│  指令集层 (isa.py)                               │
│  ISA 类：模拟硬件指令调度器                        │
│  gdma_mov2lmb, mxu_matmul, aru, ...              │
│  处理数据路径选择、地址计算、调用 semantic 执行      │
├─────────────────────────────────────────────────┤
│  语义层 (semantic.py)                            │
│  Broadcast, Binary, Unary, Reduce                │
│  纯数学函数，实现运算的数值行为                     │
│  不关心数据来自哪个硬件缓冲区                      │
├─────────────────────────────────────────────────┤
│  工具层 (utils.py + common.py)                   │
│  布局转换、对齐工具、常数定义                       │
│  M0=8, N0=8, K0=8, K0_Byte=16                   │
│  mk_to_k1mk0, k1mk0_to_mk, ...                   │
└─────────────────────────────────────────────────┘
```

### 2.2 硬件模块一览

| 模块 | 全称 | 作用 | 数据布局 |
|------|------|------|----------|
| **GM** | Global Memory | L2 全局内存，存储整个矩阵 | `k1mk0` = `(K1, M, K0)` |
| **UB** | Unified Buffer | L1 通用暂存区，存放中间结果 | `m1n1m0n0` 或 `k1mk0` |
| **LMB** | Left Matrix Buffer | 左操作数缓冲区 | `m1k1m0k0` = `(M1, K1, M0, K0)` |
| **RMB** | Right Matrix Buffer | 右操作数缓冲区 | `n1k1n0k0` = `(N1, K1, N0, K0)` |
| **PSB** | Partial Sum Buffer | MXU 输出累加器 | `m1n1m0n0` = `(M1, N1, M0, N0)` |
| **PMB** | Bias Buffer | 偏置缓冲区 | `n1n0` = `(N1, N0)` |
| **ARU** | Arithmetic Unit | 通用算术单元 | `m1n1m0n0` 输入/输出 |
| **ARB** | ARU Result Buffer | ARU 规约结果寄存器 | 1D 或 0D 标量 |
| **MXU** | Matrix Multiply Unit | 矩阵乘法引擎 | `m1k1m0k0` × `n1k1n0k0` → `m1n1m0n0` |
| **GDMA** | Global DMA | GM → LMB/RMB/PMB 数据搬运 | — |
| **LDMA** | Local DMA | UB → LMB/RMB 数据搬运 | — |

### 2.3 数据布局层级

```
L2 (全局内存): k1mk0 布局
  shape: (K1, M, K0)  其中 K1 = K // K0
  
  mk_to_k1mk0: (M, K) → (K1, M, K0)
    1. pad: (M, K) → (M, K1*K0)  [末尾补零]
    2. reshape: (M, K1*K0) → (M, K1, K0)
    3. permute: (M, K1, K0) → (K1, M, K0)
  
  k1mk0_to_mk: (K1, M, K0) → (M, k)
    1. permute: (K1, M, K0) → (M, K1, K0)
    2. reshape: (M, K1, K0) → (M, K1*K0)
    3. slice: (M, K1*K0) → (M, k)

L0 (片上): m1n1m0n0 布局
  shape: (M1, N1, M0, N0)
  其中 M1 = ceil_div(slice_m, M0), N1 = ceil_div(slice_n, N0)
  
  由 (M1*M0, N1*N0) 通过 reshape+permute 转换而来：
    (M1*M0, N1*N0) → (M1, M0, N1, N0) → (M1, N1, M0, N0)
  
  m1k1m0k0: 矩阵乘法的左操作数布局
    shape: (M1, K1, M0, K0)
    由 k1mk0 通过逐个元素搬运填充
```

### 2.4 关键常量 (common.py)

```python
M0 = 8      # M 维度最小块大小
N0 = 8      # N 维度最小块大小
K0 = 8      # K 维度最小块大小 (fp16 下对应 K0_Byte=16 字节)
C0 = N0 = 8 # C 维度与 N 相同
K0_Byte = 16
```

### 2.5 ARU 指令编码

单条 ARU 指令通过布尔标志控制所有功能，模拟真实硬件指令编码：

**数据源选择**（最多同时两路，通过布尔标志组合选择）：
```
psb_rd_en | ub_rd_en | arb_en | scalar_en | 数据源1 | 数据源2
----------|----------|--------|-----------|---------|--------
    1     |    0     |   0    |     0     |  PSB    |  None
    0     |    1     |   0    |     0     |  UB     |  None
    1     |    0     |   1    |     0     |  PSB    |  Broadcast(ARB)
    0     |    1     |   1    |     0     |  UB     |  Broadcast(ARB)
    1     |    1     |   0    |     0     |  PSB    |  UB
    0     |    1     |   0    |     1     |  UB     |  Constant(scalar)
    1     |    0     |   0    |     1     |  PSB    |  Constant(scalar)
```

**双目运算**（互斥，只能启用一个）：
```
add_en | sub_en | max_en | min_en | mul_en | div_en |  效果
-------|--------|--------|--------|--------|--------|----------
   1   |   0    |   0    |   0    |   0    |   0    | x1 + x2
   0   |   1    |   0    |   0    |   0    |   0    | x1 - x2
   0   |   0    |   1    |   0    |   0    |   0    | max(x1, x2)
   0   |   0    |   0    |   1    |   0    |   0    | min(x1, x2)
   0   |   0    |   0    |   0    |   1    |   0    | x1 * x2
   0   |   0    |   0    |   0    |   0    |   1    | x1 / x2
   0   |   0    |   0    |   0    |   0    |   0    | 无双目运算
```

**单目运算**（按顺序链式执行）：
```
neg → clamp → exp → sqrt → pow → recp
```
若某标志为 False，则跳过该步骤。

**规约运算**：
```
reduce_m_en | reduce_n_en | reduce_mode |  效果
------------|-------------|-------------|----------
    1       |     0       |      0      | max over M → (N,) 1D
    0       |     1       |      0      | max over N → (M,) 1D
    1       |     0       |      1      | min over M → (N,) 1D
    0       |     1       |      1      | min over N → (M,) 1D
    1       |     0       |      2      | sum over M → (N,) 1D
    0       |     1       |      2      | sum over N → (M,) 1D
    1       |     0       |      3      | mean over M → (N,) 1D
    0       |     1       |      3      | mean over N → (M,) 1D
```

**写回目标**：
```
ub_wr_en | ub_layout | gm_wr_en | arb_wr_en |  效果
---------|-----------|----------|-----------|----------
    1    |     0     |    0     |     0     | 写入 UB，保持 m1n1m0n0 布局
    1    |     1     |    0     |     0     | 写入 UB，转为 k1mk0 布局（最终输出格式）
    0    |     -     |    1     |     0     | 写入 GM
    0    |     -     |    0     |     1     | 写入 ARB（规约结果，通常 1D）
    1    |     0     |    0     |     1     | 同时写入 UB + ARB（ub_wr_en 优先 append 到返回值）
    0    |     -     |    0     |     0     | 无写回
```

---

## 3. Correct `op_softmax` Walkthrough

### 3.1 分步执行流程

```python
def op_softmax(matrix_mn):
    M_L2 = matrix_mn.shape[0]        # 总行数，如 67
    N_L2 = matrix_mn.shape[1]        # 总列数，如 45
    N1_L2 = ceil_div(N_L2, N0)       # N1 块数，如 ceil_div(45,8) = 6
    M_L0 = np.random.randint(3, M_L2) # 随机 tile 高度，如 14
    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)  # (M,N) → (K1,M,K0), 这里 K≡N
    result_mn = torch.zeros((M_L2, N_L2), dtype=matrix_mn.dtype)  # 预分配输出
```

**关键点**：softmax 在 N 维规约，所以借用 `mk_to_k1mk0` 的 K1-K0 框架处理 N 维度——即把 N 当作 K 来对齐。

### 3.2 M 维度分块循环

```python
    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
```

对于每个 tile，处理 `m_size_l0` 行（最后一块可能较小）。

### 3.3 数据搬运：GM → PSB（借用 LMB 路径）

```python
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(
            matrix_n1mn0_l2, M_L2, l0_m_start_in_l2,
            N1_L2, 0, m_size_l0, N1_L2)
```

**为什么用 `gdma_mov2lmb` 而不是专门的 GM→PSB 指令？**
因为目前没有 GM→PSB 的直接数据通路。`gdma_mov2lmb` 从 GM 的 `k1mk0` 布局读取数据，填充到 `m1k1m0k0` 布局。这里将 N 维度映射为 K 维度，结果得到 `m1n1m0n0` 形状。

**流程**：
```
GM: matrix_n1mn0_l2  shape: (N1, M, N0)  其中 N1 = ceil_div(N, 8)
          ↓  gdma_mov2lmb
PSB: matrix_m1n1m0n0_l0  shape: (M1, N1, M0, N0)
     M1 = ceil_div(m_size_l0, 8), N1 = ceil_div(N_L2, 8)
```

### 3.4 ARU 指令 1：求行最大值 (max reduce over N)

```python
        max_m1n1m0n0_arb, = isa.aru(
            slice_m=m_size_l0, slice_n=N_L2,
            psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True,
            ub_m1n1m0n0=None, ub_rd_en=False,
            arb_in=None, arb_en=False, br_m=False, br_n=False,
            scalar_en=False, scalar=None,
            add_en=False, sub_en=False, max_en=False, min_en=False,
            mul_en=False, div_en=False, neg_en=False, clamp_en=False,
            clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
            pow_en=False, recp_en=False,
            reduce_m_en=False, reduce_n_en=True, reduce_mode=0,  # max over N
            ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
        )
```

**执行路径**：
```
Data source selection:
  psb_rd_en=True, ub_rd_en=False, arb_en=False, scalar_en=False
  → matches: x1 = psb_m1n1m0n0, x2 = None

Binary: all False → skip, x = x1

Unary: all False → skip, x unchanged

Reduce:
  reduce_m_en=False, reduce_n_en=True, reduce_mode=0
  → Reduce(x, slice_m, slice_n, False, True, 0)
    → reshape m1n1m0n0 to (M1*M0, N1*N0)
    → clip to [:slice_m, :slice_n]
    → torch.max(x_mn, dim=1).values  → shape (M,) 1D vector

Write-back:
  ub_wr_en=False, gm_wr_en=False, arb_wr_en=True
  → write reduced 1D vector to ARB
  → return_value = [(1D vector in ARB)]
```

**输出**：`max_m1n1m0n0_arb` 是一个形状为 `(m_size_l0,)` 的 1D 向量，每行的最大值。

### 3.5 ARU 指令 2：减最大值 + exp + sum reduce

```python
        exp_m1n1m0n0_ub, sum_arb, = isa.aru(
            m_size_l0, N_L2,
            psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True,
            ub_m1n1m0n0=None, ub_rd_en=False,
            arb_in=max_m1n1m0n0_arb, arb_en=True,
            br_m=False, br_n=True, scalar_en=False, scalar=None,
            add_en=False, sub_en=True, max_en=False, min_en=False,
            mul_en=False, div_en=False, neg_en=False, clamp_en=False,
            clamp_min=None, clamp_max=None, exp_en=True, sqrt_en=False,
            pow_en=False, recp_en=False,
            reduce_m_en=False, reduce_n_en=True, reduce_mode=2,  # sum over N
            ub_wr_en=True, ub_layout=0, gm_wr_en=False, arb_wr_en=True
        )
```

**执行路径**：
```
Data source selection:
  psb_rd_en=True, ub_rd_en=False, arb_en=True, scalar_en=False
  → matches: x1 = psb_m1n1m0n0, x2 = Broadcast(arb_in, slice_m, slice_n, br_m=False, br_n=True)

Broadcast:
  arb_in shape: (m_size_l0,)  1D
  br_m=False, br_n=True
  → x2_mn = arb_in.unsqueeze(1).expand(slice_m, slice_n)
  → reshape+permute to m1n1m0n0 layout (M1, N1, M0, N0)
  → broadcasts each row's max across all N columns

Binary:
  sub_en=True → x = x1 - x2 = (original_value - row_max)

Unary:
  exp_en=True → x = torch.exp(x)   # e^(x - max)

Reduce:
  reduce_n_en=True, reduce_mode=2 → sum over N dim
  → shape (M,) 1D vector, each element is sum of exp for that row

Write-back:
  ub_wr_en=True → return_value[0] = x (the exp values, kept as m1n1m0n0)
  arb_wr_en=True → x goes through Reduce first, then return_value[1] = reduced 1D vector
```

**输出**：
- `exp_m1n1m0n0_ub`：exp 结果，形状 `(M1, N1, M0, N0)`，写入 UB，后续还需使用
- `sum_arb`：每行的 exp 之和，形状 `(m_size_l0,)`，写入 ARB

### 3.6 ARU 指令 3：除法归一化

```python
        result_n1mn0, = isa.aru(
            m_size_l0, N_L2,
            psb_m1n1m0n0=None, psb_rd_en=False,
            ub_m1n1m0n0=exp_m1n1m0n0_ub, ub_rd_en=True,
            arb_in=sum_arb, arb_en=True,
            br_m=False, br_n=True, scalar_en=False, scalar=None,
            add_en=False, sub_en=False, max_en=False, min_en=False,
            mul_en=False, div_en=True, neg_en=False, clamp_en=False,
            clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
            pow_en=False, recp_en=False,
            reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
            ub_wr_en=True, ub_layout=1, gm_wr_en=False, arb_wr_en=False
        )
```

**执行路径**：
```
Data source selection:
  psb_rd_en=False, ub_rd_en=True, arb_en=True, scalar_en=False
  → matches: x1 = ub_m1n1m0n0 (exp values), x2 = Broadcast(arb_in, ...)

Broadcast:
  arb_in = sum_arb, shape (m_size_l0,)
  br_n=True → broadcast each row's sum across all N columns
  → x2 shape (M1, N1, M0, N0), each row filled with that row's sum

Binary:
  div_en=True → x = x1 / x2 = exp / sum(exp)

Unary: all False → skip

Reduce: reduce_m_en=False, reduce_n_en=False → skip, x unchanged

Write-back:
  ub_wr_en=True, ub_layout=1
  → layout conversion: m1k1m0k0_to_k1mk0(x, slice_m)
     (M1, N1, M0, N0) → (N1, M, N0)  i.e. k1mk0 layout
  → return_value[0] = result in k1mk0 layout
```

**输出**：`result_n1mn0`，形状 `(N1, m_size_l0, N0)` 即 k1mk0 布局的 softmax 结果。

### 3.7 写回结果矩阵 + 循环收尾

```python
        result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, :] = \
            k1mk0_to_mk(result_n1mn0, N_L2)
    return result_mn    # ← 在循环外！这是 Bug #3 的修复点
```

`k1mk0_to_mk` 将 `(N1, m_size_l0, N0)` → `(m_size_l0, N_L2)`，然后写入预分配的 `result_mn` 对应行。

### 3.8 完整数据流图

```
Input: matrix_mn (M_L2, N_L2)
  │
  ├─ mk_to_k1mk0()  →  matrix_n1mn0_l2 (N1, M_L2, N0)
  │
  └─ for each M tile ─────────────────────────────────────┐
     │                                                     │
     │  gdma_mov2lmb                                      │
     │  GM (n1mn0) → PSB (m1n1m0n0)                       │
     │                                                     │
     │  ┌─ ARU #1 ──────────────────────────┐              │
     │  │  PSB → (无Binary) → (无Unary)      │              │
     │  │  → Reduce(max, N)                  │              │
     │  │  → ARB: max_row (M,)               │              │
     │  └────────────────────────────────────┘              │
     │                                                     │
     │  ┌─ ARU #2 ──────────────────────────┐              │
     │  │  PSB - Broadcast(ARB) → sub       │              │
     │  │  → exp → Reduce(sum, N)            │              │
     │  │  → UB: exp(m1n1m0n0)              │              │
     │  │  → ARB: sum_exp (M,)               │              │
     │  └────────────────────────────────────┘              │
     │                                                     │
     │  ┌─ ARU #3 ──────────────────────────┐              │
     │  │  UB / Broadcast(ARB) → div        │              │
     │  │  → UB(layout=1): result(n1mn0)    │              │
     │  └────────────────────────────────────┘              │
     │                                                     │
     │  k1mk0_to_mk(result_n1mn0, N_L2)                    │
     │  → (m_size_l0, N_L2) → write to result_mn[i:i+m]   │
     │                                                     │
     └─────────────────────────────────────────────────────┘
     
Output: result_mn (M_L2, N_L2)
```

---

## 4. Key Truth Tables

### 4.1 ARU 数据源选择真值表

| psb_rd_en | ub_rd_en | arb_en | scalar_en | x1 来源 | x2 来源 | 典型用途 |
|-----------|----------|--------|-----------|---------|---------|----------|
| 1 | 0 | 0 | 0 | PSB | None | 对 PSB 数据做单目运算 |
| 0 | 1 | 0 | 0 | UB | None | 对 UB 数据做单目运算 |
| 1 | 0 | 1 | 0 | PSB | Broadcast(ARB) | PSB 对广播值做双目运算 |
| 0 | 1 | 1 | 0 | UB | Broadcast(ARB) | UB 对广播值做双目运算 |
| 1 | 1 | 0 | 0 | PSB | UB | PSB 和 UB 做双目运算 |
| 0 | 1 | 0 | 1 | UB | Constant(scalar) | UB 对标量做双目运算 |
| 1 | 0 | 0 | 1 | PSB | Constant(scalar) | PSB 对标量做双目运算 |
| 0 | 0 | 0 | 0 | — | — | **invalid** (无数据源) |
| * | * | * | * (>1 True) | — | — | **invalid** (超过两个源) |

### 4.2 Broadcast 操作真值表

| br_m | br_n | ARB 输入形状 | 输出形状 | 语义 |
|------|------|-------------|----------|------|
| True | True | 标量 () | (M1, N1, M0, N0) | 全矩阵广播 |
| True | False | (N,) 1D | (M1, N1, M0, N0) | 沿 M 广播行 |
| False | True | (M,) 1D | (M1, N1, M0, N0) | 沿 N 广播列 |
| False | False | — | — | **assert 失败** |

在 softmax 中使用 `br_m=False, br_n=True`：
- ARU #2：`(M,)` 的行最大值广播到所有 N 列
- ARU #3：`(M,)` 的行指数和广播到所有 N 列

### 4.3 Reduce 操作真值表

| reduce_mode | reduce_n_en=True | reduce_m_en=True |
|-------------|-----------------|-----------------|
| 0 (max) | max over N → (M,) | max over M → (N,) |
| 1 (min) | min over N → (M,) | min over M → (N,) |
| 2 (sum) | sum over N → (M,) | sum over M → (N,) |
| 3 (mean) | mean over N → (M,) | mean over M → (N,) |

**Reduce 的正确实现（Bug #4 修复后）**：
```python
def Reduce(x, slice_m, slice_n, reduce_m_en, reduce_n_en, reduce_mode):
    M1, N1, M0, N0_val = x.shape
    x_mn = x.permute(0, 2, 1, 3).reshape(M1 * M0, N1 * N0_val)
    x_mn = x_mn[:slice_m, :slice_n]  # ← 关键：clip padding 区域
    # 然后按 reduce_mode 执行规约
```

**为什么需要 clip？**
- 输入 x 形状为 `(M1, N1, M0, N0)`，其中 `M1 = ceil_div(slice_m, M0)`, `N1 = ceil_div(slice_n, N0)`
- 当 `slice_m` 不是 M0 的倍数或 `slice_n` 不是 N0 的倍数时，reshape 后的 `(M1*M0, N1*N0)` 矩阵中含有 padding 的零值
- 零值会影响 sum/mean 结果的正确性（对 max/min 结果也有影响）
- 因此必须先 `[:slice_m, :slice_n]` 截取有效区域再执行 reduce

### 4.4 写回布局真值表 (ub_layout)

| ub_layout | 写入 UB 的布局 | 转换函数 | 用途 |
|-----------|---------------|---------|------|
| 0 | `m1n1m0n0` (4D) | 不变 | 中间计算结果，后续仍需在 L0 运算 |
| 1 | `k1mk0` (3D) | `m1k1m0k0_to_k1mk0(x, slice_m)` | 最终输出，准备写回 L2 或 GM |

在 softmax 中：
- ARU #2：`ub_layout=0`，exp 保留为 `m1n1m0n0` 便于 ARU #3 直接读取
- ARU #3：`ub_layout=1`，转为 `n1mn0` (k1mk0 格式) 准备写回全局内存

### 4.5 单目运算执行顺序

```
输入 x 按以下顺序链式处理：
  if neg_en:   x = -x
  if clamp_en: x = clamp(x, clamp_min, clamp_max)
  if exp_en:   x = exp(x)
  if sqrt_en:  x = sqrt(x)
  if pow_en:   x = x^2
  if recp_en:  x = 1/x
```

可以同时启用多个单目运算。例如 `neg_en=True, exp_en=True` 先取负后取指数得到 `exp(-x)`（sigmoid 中用到）。

### 4.6 返回值规则

`isa.aru()` 返回一个 tuple，顺序规则：
1. 如果 `ub_wr_en=True`：第一个元素为 UB 写入数据（如果 layout=1 已转为 k1mk0）
2. 如果 `gm_wr_en=True`：下一个元素为 GM 写入数据
3. 如果 `arb_wr_en=True`：最后一个元素为 ARB 写入数据（若有 reduce，则为 reduce 后的 1D 结果）

---

## 5. Role of Each Module in Softmax

### 5.1 GM (Global Memory) — 全局存储器

**角色**：存放完整的输入矩阵和输出矩阵

**在 softmax 中**：
- 输入 `matrix_mn` 经过 `mk_to_k1mk0()` 转换为 `(N1, M, N0)` 布局后存放在 GM
- GM 中的数据块通过 `gdma_mov2lmb` 指令按 tile 搬运到 L0
- 最终结果由 `k1mk0_to_mk()` 转回 `(M, N)` 后存放在 `result_mn`（Python 变量，逻辑上代表 GM 区域）

**为什么 softmax 的 GM 布局是 `(N1, M, N0)` 而不是 `(K1, M, K0)`？**
因为 softmax 需要沿 N 维规约，所以 N 维度被映射为 K 维度来利用 K0=8 的对齐机制。在 LM 视角中，N → K。

### 5.2 GDMA (Global DMA) — 全局数据搬运引擎

**角色**：将 GM 中的 `k1mk0` 布局数据按照 tile 坐标搬运到 LMB（左矩阵缓冲区）

**在 softmax 中**：
```python
matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(
    matrix_n1mn0_l2,    # GM 数据源
    M_L2,               # GM 的总 M
    l0_m_start_in_l2,   # 当前 tile 在 M 维的起始偏移
    N1_L2,              # GM 的总 N1 块数
    0,                  # N1 起始偏移（softmax 不切分 N）
    m_size_l0,          # 要搬运的 M 行数
    N1_L2               # 要搬运的 N1 块数（全量）
)
```

`gdma_mov2lmb` 内部逻辑：
1. 分配 `(M1, K1, M0, K0)` 零张量作为 LMB
2. 三层循环遍历 `(m1, k1, m0)` 坐标
3. 计算全局坐标 `(start_k1 + k1, start_m + m1*M0 + m0)`
4. 从 GM 的 `k1mk0` 中按行拷贝 K0=8 个元素

**关键**：gdma_mov2lmb 的目标是 LMB，但 softmax 中我们将其数据导入 PSB 作为 ARU 的输入（因为目前缺乏 GM→PSB 直连通路）。

### 5.3 LMB (Left Matrix Buffer) — 左矩阵缓冲区

**角色**：存放 MXU 的左操作数 `m1k1m0k0`

**在 softmax 中**：
- `gdma_mov2lmb` 的返回值直接传入 ARU 的 `psb_m1n1m0n0` 参数
- 虽然名称是 "LMB"，但在 softmax 场景中它充当了从 GM 到 PSB 的数据中转站
- 数据实际上变成了 `(M1, N1, M0, N0)` 而非标准的 `(M1, K1, M0, K0)`

### 5.4 PSB (Partial Sum Buffer) — 部分和缓冲区

**角色**：存放 MXU 的矩阵乘法输出，也是 ARU 的数据源之一

**在 softmax 中**：
- PSB 是 softmax 的**主数据源**——存放当前 tile 的输入数据
- 三条 ARU 指令中，ARU #1 和 ARU #2 都从 PSB 读取输入数据 (`psb_rd_en=True`)
- ARU #3 不再从 PSB 读（`psb_rd_en=False`），因为此时需要的是 exp 结果

### 5.5 ARU (Arithmetic Unit) — 算术运算单元

**角色**：执行所有非矩阵乘法的运算——双目、单目、规约、广播、布局转换

**在 softmax 中的三次调用**：

| 调用 | 双目运算 | 单目运算 | 规约 | 数据源 | 写回目标 | 作用 |
|------|---------|---------|------|--------|---------|------|
| ARU #1 | 无 | 无 | max over N | PSB | ARB | 求每行最大值 |
| ARU #2 | sub (PSB - max) | exp | sum over N | PSB + ARB | UB + ARB | 减最大值+取指数+求指数和 |
| ARU #3 | div (exp / sum) | 无 | 无 | UB + ARB | UB(layout=1) | 除以指数和得到概率 |

**ARU 内部数据流**：
```
输入 ─→ 数据源选择 (x1, x2) ─→ Binary(x1, x2) ─→ Unary ─→ Reduce ─→ 写回
                │                     │              │          │
                │               Broadcast(ARB)    neg/clamp      │
                │               (若 arb_en=True)  /exp/sqrt     │
                │                                 /pow/recp     │
                │                                               │
                └── PSB / UB / scalar ──────────────────────────┘
```

### 5.6 UB (Unified Buffer) — 统一缓冲区

**角色**：L1 级通用暂存区，存放 ARU 运算的中间结果

**在 softmax 中**：
- ARU #2 将 exp 结果写入 UB（`ub_wr_en=True, ub_layout=0`），保持 `m1n1m0n0` 布局
- ARU #3 从 UB 读取 exp 结果（`ub_rd_en=True`），与广播的 sum 做除法
- ARU #3 将最终结果写入 UB（`ub_wr_en=True, ub_layout=1`），转为 `k1mk0` 布局

**UB 的两种布局模式**：
- `ub_layout=0`：保持 `m1n1m0n0` — 用于中间计算，后续指令可直接读取
- `ub_layout=1`：转为 `k1mk0` — 用于最终输出，可直接写回全局内存

### 5.7 ARB (ARU Result Buffer) — 算术结果缓冲区

**角色**：存放 ARU 的规约结果（1D 向量或标量），并提供广播数据源

**在 softmax 中**：
- ARU #1 → ARB：`max_row`，形状 `(m_size_l0,)`，每行的最大值
- ARU #2 → ARB：`sum_exp`，形状 `(m_size_l0,)`，每行的指数和
  - 注意：ARU #2 同时写入 UB（中间结果 exp）和 ARB（规约结果 sum）
- ARU #3：`arb_en=True` 读取 ARU #2 的 sum_arb，通过 `Broadcast(..., br_n=True)` 广播到所有 N 列

**ARB 的关键特性**：
- 存放规约后的紧凑数据（远小于完整 tile）
- 作为 Broadcast 的数据源，将小向量扩展回完整 tile 尺寸
- 这是硬件设计中的关键优化：避免从 UB 回读规约结果

### 5.8 MXU (Matrix Multiply Unit) — 矩阵乘法单元

**角色**：执行 `m1k1m0k0` × `n1k1n0k0` → `m1n1m0n0` 的矩阵乘法

**在 softmax 中**：**不使用 MXU**。softmax 是纯逐元素/规约运算，完全由 ARU 完成。

### 5.9 RMB (Right Matrix Buffer) / PMB (Bias Buffer)

**在 softmax 中**：均不使用。RMB 存放右操作数，PMB 存放偏置，都是 MXU 的配套模块。

---

## Appendix: The 5 Bugs Fixed (Summary)

| Bug # | Location | Symptom | Root Cause | Fix |
|-------|----------|---------|------------|-----|
| 1 | `activation.py` ARU #2 | `UnboundLocalError` | `ub_m1n1m0n0=exp_m1n1m0n0_ub` 在变量定义前引用自身 | `ub_m1n1m0n0=None, ub_rd_en=False` |
| 2 | `activation.py` ARU #3 | 错误结果 | 从 PSB 读原始数据 + 用 sub+exp+reduce 而非 div | `psb_rd_en=False, ub_rd_en=True, div_en=True` |
| 3 | `activation.py` | 只处理第一个 tile | `return result_mn` 在 for 循环内部 | 移到循环外，预分配完整结果矩阵 |
| 4 | `semantic.py` Reduce() | 规约结果偏大/不正确 | padding 的零值参与 sum/mean 计算 | 在 reshape 后添加 `[:slice_m, :slice_n]` 切片 |
| 5 | `isa.py` ARU Reduce 调用 | TypeError | `Reduce()` 调用缺少 `slice_m/slice_n` 参数 | 传递 `slice_m, slice_n` 到 `Reduce()` |