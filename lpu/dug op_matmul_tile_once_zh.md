# op_matmul_tile_once 调试总结

## 范围

本文档记录了以下函数的调试与修复工作：

```python
op_matmul_tile_once(left, right, bias)
```

期望的计算结果为：

```python
torch.matmul(left, right.permute(1, 0)) + bias.unsqueeze(0)
```

其中 `left` 形状为 `(M, K)`，`right` 形状为 `(N, K)`，`bias` 形状为 `(N,)`。

## 首次失败

首次失败是：

```text
TypeError: aru() got an unexpected keyword argument 'arb_out'
```

`ISA.aru` 未定义 `arb_out` 参数。因此问题出在 `op_matmul_tile_once` 中过时的调用点，而非 `aru` 实现不完整。

## 修复列表

### 1. 更新 `op_matmul_tile_once` 中的 ARU 调用

该调用现在提供当前 `aru` 接口所需的参数：

- `scalar_en=False`
- `scalar=None`
- `ub_layout=1`
- `arb_wr_en=False`

同时移除了不支持的 `arb_out` 参数。

`aru` 返回一个已启用的输出缓冲区列表，因此单个结果通过以下方式解包：

```python
result_n1mn0, = isa.aru(...)
```

### 2. 使用正确的右矩阵传输指令

右操作数的数据布局为 `(k1, n, k0)`，因此必须使用以下指令加载：

```python
isa.gdma_mov2rmb(...)
```

使用 `gdma_mov2lmb` 会错误地将其 N 维度当作 M 维度处理。

### 3. 保留所有 L0 输出分块

原来的实现只保留了最后一个 L0 输出分块。修复后的代码创建了一个完整的 `(M, N)` 结果张量，并将每个完成的分块写回其全局坐标范围：

```python
result_mn[m_start:m_start + m_size, n_start:n_start + n_size] = tile_result
```

这支持多个 M 和 N 分块，包括小于 `M_L0` 或 `N_L0` 的尾部块。

### 4. 选择浮点 MXU 模式

测试生成的张量为浮点数。`ISA.mxu_matmul` 默认使用 `dtype='int8'`，这会在乘法之前截断浮点输入，导致较大的数值误差。

现在矩阵乘法调用显式使用：

```python
dtype='fp16'
```

### 5. 修正 ISA 传输中的 K1 索引

`ISA.gdma_mov2lmb` 和 `ISA.gdma_mov2rmb` 接收的 `start_tensor_k1` 和 `k1` 以 K1 为单位。其 GM 访问之前额外乘了一个 `* K0`：

```python
# 错误
gm_k1mk0[start_tensor_k1 + k1 * K0, ...]
```

正确的访问方式是：

```python
gm_k1mk0[start_tensor_k1 + k1, ...]
```

当 K 有多个 K1 块时，多余的乘法会导致 `IndexError`。

### 6. 修正 ARU 输出布局转换

`m1k1m0k0_to_k1mk0` 将分块数据从 `(m1, k1, m0, k0)` 转换为 `(k1, m, k0)`。其局部变量 `M1` 和 `K1` 的赋值是颠倒的，导致输出行数错误。

修正后的维度赋值为：

```python
M1 = m1k1m0k0.shape[0]
K1 = m1k1m0k0.shape[1]
```

对于 ARU 输出，同样的布局映射得到 `(n1, m, n0)`。

### 7. 允许最小尺寸的 L1 分块

`find_optimal_l0_tiling` 原来使用 `np.random.randint(2, M_L1)`。如果 L1 分块大小为 2，这就会变成 `np.random.randint(2, 2)`，导致以下错误：

```text
ValueError: low >= high
```

上界现在为包含性上界，例如：

```python
np.random.randint(2, M_L1 + 1)
```

## 问题归属总结

| 问题 | 归属位置 |
| --- | --- |
| 过时的 `aru` 参数、RMB 选择、分块组装、MXU dtype | `ops/matmul.py` |
| GM 到缓冲区传输中的 K1 地址计算 | `isa.py` |
| 分块布局维度转换 | `utils.py` |
| 无效的随机 L0 分块边界 | `ops/matmul.py` |

## 验证

目标测试已成功运行：

```powershell
python -m unittest test.BlockTestMatmul.test_matmul_tile_once
```

随后随机测试重复了十次，覆盖了 30 个生成的 `(M, N, K)` 配置。全部运行通过，观察到的平均误差率低于测试阈值。