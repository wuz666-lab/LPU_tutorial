# Deconvolution HLS Line Buffer 时序说明

本文描述 [`hls/include/deconv.h`](hls/include/deconv.h) 中 `deconv()` 的数据流。

> **关于 cycle 的说明**：`deconv()` 当前没有 `#pragma HLS PIPELINE`，因此 C++/C-sim
> 的循环顺序是功能语义，综合工具并不保证本文表格中的固定周期数。下文的 cycle 表把
> 每次最内层 `fi` 迭代抽象为一个目标 `II=1` 的逻辑事务，用于定义 RTL/HLS 实现时的
> RAM 写读、ROM 取权重、MAC 和输出事件。若 line buffer/weight 被映射为同步 RAM/ROM，
> RAM/ROM 返回的数据和 `valid/first/last` 标志必须额外延迟一拍后再送入 MAC。

## 细粒度调度

大 padding 会使 `N_OH/N_OW` 缩小，因而旧的“由固定 `ow` 点预取一列”的策略可能无法写满下一 IFM 行。当前实现统一使用三个写指针 ih_to_read / iw_to_read / fi_to_read 表示输入流的下一个 beat；每次写入只推进一个 beat。主循环在 ow=0 前确保该坐标所需的最高行/列已就绪，随后每完成一个 ow 预取一个完整输入像素（全部 fi），在 ow 末尾把正在预取的行补齐。

1. 启动阶段将 `oh=0` 需要的 IFM 行完整写入。
2. 计算任一 `oh` 的各个 `ow` 时，在第一个 kernel 的 `fi` 循环中交错写入下一 IFM 行的一个输入像素。
3. 切换到新的 `oh` 前，仅检查新行中 `ow=0` 会访问到的最大 IFM 行/列；未写到该位置才暂停计算并继续输入。
4. 每个新的 `ow` 前只检查该 `ow` 需要的最大 IFM 列；`ow=N_OW-1` 后若当前预取行未写满，则暂停并写满该行，再进入下一个 `oh`。

通常情况下计算与输入按像素交错；只有输出行结束或下一个输出坐标尚未 ready 时才产生补写停顿。

## 1. Line buffer 容量

```text
RESIDENT_ROWS = floor((K - 1) / S) + 1 = ceil(K / S)
LB_H = min(RESIDENT_ROWS + 1, N_IH)
```

`RESIDENT_ROWS` 是一个输出行可能同时访问的最大 IFM 行数。`+1` 是正常的逐像素
prefetch 槽：计算当前输出时可以写入下一 IFM 行，而不会覆盖仍会被当前输出读取的行。
当 `N_IH` 较小，`LB_H` 截断到 `N_IH`，避免 line buffer 大于完整 IFM 高度。

## 2. 整体数据流

```mermaid
flowchart LR
	I[IFM stream] --> S[启动控制器]
	S --> LB[Line buffer\nLB_H x N_IW x FOLD_I]
	O[oh/ow/fo/kh/kw/fi 控制器] --> R[坐标映射\nih=(oh-kh+P)/S\niw=(ow-kw+P)/S]
	R -->|坐标合法| LB
	O --> W[Weight array / ROM]
	LB --> M[并行 P_ICH MAC]
	W --> M
	M --> A[P_OCH accumulators]
	A -->|kh=0, kw=0, fi=FOLD_I-1| OFM[OFM stream]
	O --> D[尾部排空控制]
	D --> I
```

### 3.1 启动预加载

启动控制器只针对 `oh=0` 计算最大可访问行：

```text
h_init = P - kh, kh = 0 .. K - 1
max_ih_init = max((P - kh) / S)
```

计算时仅保留满足 `h_init >= 0`、`h_init % S == 0`、且 `0 <= ih < N_IH` 的项。
随后从 `ih_to_read=0` 起连续读取完整 IFM 行，直至 `ih_to_read > max_ih_init`：

```text
for ih = 0 .. max_ih_init:
	for iw = 0 .. N_IW - 1:
		for fi = 0 .. FOLD_I - 1:
			line_buf[ih % LB_H][iw][fi] = in.read()
```

所以大 padding 不会导致首个输出访问尚未写入的高行。例如 `P=3, K=2, S=1, oh=0`：

```text
kh=1 -> ih=(0-1+3)/1=2
kh=0 -> ih=(0-0+3)/1=3
```

启动阶段会读入 IFM 行 `0..3`，环形 buffer 中仍保留计算所需的行 `2,3`。

### 3.2 计算期间的细粒度写入

当前实现统一维护以下输入写指针：

```text
ih_to_read : 下一个输入 beat 所属的绝对 IFM 行
iw_to_read : 该行中下一个待写入的 IFM 列
fi_to_read : 该列中下一个待写入的输入通道 fold
```

每调用一次 `deconv_write_input_beat()` 都从 `in` 消费一个 beat 并执行：

```text
line_buf[ih_to_read % LB_H][iw_to_read][fi_to_read] = in.read()
fi_to_read++
fi_to_read == FOLD_I -> fi_to_read=0, iw_to_read++
iw_to_read == N_IW  -> iw_to_read=0, ih_to_read++
```

对当前输出坐标 `(oh,ow)`，定义：

```text
max_ih(oh) = max valid ((oh - kh + P) / S)
max_iw(ow) = max valid ((ow - kw + P) / S)
```

其中 `valid` 表示整除、非负、且落在对应 IFM 高度/宽度内。具体调度如下：

| 时机 | ready 条件与动作 |
|---|---|
| 进入 `oh` | 若 `ih_to_read` 尚未越过 `max_ih(oh)`，或正写 `max_ih(oh)` 但 `iw_to_read <= max_iw(0)`，持续写入直至 `ow=0` 所需数据就绪。 |
| 进入每个 `ow` | 同样检查 `max_iw(ow)`；只补写到该 `ow` 需要的最大列，不预先补齐整行。 |
| 计算当前 `ow` | 在 `fo=0 && kh=K-1 && kw=K-1` 的 `fi` 循环里，每个 `fi` 写入下一 IFM 像素的一个 fold；因此写入与 MAC 内层事务交错。 |
| `ow=N_OW-1` | 若仍在预取下一 `oh` 所需的 IFM 行，将该行剩余 `iw/fi` 全部写完，再开始下一个 `oh` 的 `ow=0` ready 检查。 |

这套规则解决了大 padding 下 `N_OW` 太小的问题：不会假定一行 OFM 的固定数量输出能恰好
填满一行 IFM；缺少的列仅在行尾按需补齐。

### 3.3 RAM 读、ROM 读和 MAC

对每个 `Q=(oh,ow,fo,kh,kw,fi)`：

1. 始终请求/读取 `weight[fo][fi][kh*K+kw]`。
2. 仅当坐标合法时读取 `line_buf[ih % LB_H][iw][fi]`。
3. 将该 IFM 向量拆成 `P_ICH` 个 `x`，将权重向量拆成每个 `poc` 对应的 `P_ICH` 个 `w`。
4. 对每个 `poc` 执行：`acc[poc] += sum(pic: x[pic] * w[poc][pic])`。
5. 每一个 `(oh,ow,fo)` 开始前，`acc[0..P_OCH-1]` 清零。

坐标无效时没有 IFM MAC 更新，但内层循环仍会继续推进到下一个 `kh/kw/fi`，以保持权重
与控制计数器的固定顺序。

### 3.4 输出有效

HLS 中没有显式 `out_valid` 信号；逻辑上的 output-valid 事件是：

```text
kh == 0 && kw == 0 && fi == FOLD_I - 1
```

在该事件中，`out.write(out_buf)` 写出一个包含 `P_OCH` 个结果的输出 beat，即
`OFM[oh][ow][fo]`。由于 `acc` 已在该输出块开始前清零，且全部有效 MAC 已完成，这个
beat 对应完整的累加结果。

若映射到同步 RAM/ROM 的 RTL，最后一次读的返回与 MAC 累加发生在请求后一拍；因此 RTL
的 `out_valid` 应再相对最后一次 MAC 对齐/延迟，而不能直接使用未延迟的循环末尾标志。

### 3.5 尾部排空

所有 `oh/ow/fo` 计算完成后，可能仍有 `ih_to_read .. N_IH-1` 的 IFM 行没有被任何输出
访问。此时不再进行 MAC，但继续调用同一个细粒度写入器，保持当前可能未完成的
`iw_to_read/fi_to_read` 位置，直至所有输入 beat 都被消费：

```text
while ih_to_read < N_IH:
	deconv_write_input_beat(in, line_buf,
	                        ih_to_read, iw_to_read, fi_to_read)
```

该排空阶段保证 `assert(in.empty())` 成立，并保证上游 stream 不会因未消费的 IFM 停住。

## 4. 例子：K=3, P=1, S=2

本例采用现有 `tb_deconv.cpp` 的参数：

```text
N_IH=N_IW=5, N_ICH=4, N_OCH=6
P_ICH=2, P_OCH=3, FOLD_I=2, FOLD_O=2
K=3, P=1, S=2, O_P=1
N_OH=N_OW=9, RESIDENT_ROWS=2, LB_H=3
```

对 `oh=0`：只有 `kh=1` 满足高度坐标条件，得到 `ih=0`，所以启动阶段读 IFM 第 0 行：

```text
启动输入 beat 数 = 1 * N_IW * FOLD_I = 10
RAM 地址范围 = 0 .. 9
ih_to_read 在启动结束后为 1
```

随后 `oh=0, ow=0, fo=0` 的 kernel 访问按 `kk=8..0`、每个 `kk` 两个 `fi` 进行。
有效坐标仅为 `(kh,kw)=(1,1)`，即 `kk=4`，并从 `IFM[0][0]` 读两个输入 fold。

```text
// cycle  | oh ow | cntr_fo | cntr_fi | cntr_kk | in_data / RAM 写入       | RAM_WE | RAM_ADDR | ROM_ADDR | MAC 输入                 | acc 状态                         | out_valid
// -------|-------|---------|---------|---------|--------------------------|--------|----------|----------|--------------------------|----------------------------------|----------
//   1    | init  |    -    |    0    |    -    | IFM[0][0], ich[0:1]      |   1    |    0     |    -     | 无                       | 无                               |    0
//   2    | init  |    -    |    1    |    -    | IFM[0][0], ich[2:3]      |   1    |    1     |    -     | 无                       | 无                               |    0
//  ...   | init  |    -    |   ...   |    -    | IFM[0][1..4], 两个 fold  |   1    |  2..9    |    -     | 无                       | 无                               |    0
// -------|-------|---------|---------|---------|--------------------------|--------|----------|----------|--------------------------|----------------------------------|----------
//  11    | 0  0 |    0    |    0    |    8    | IFM[1][0], ich[0:1]      |   1    |   10     |    8     | 无，(kh,kw)=(2,2) 无效   | acc 已在本 OFM 块开始时清零      |    0
//  12    | 0  0 |    0    |    1    |    8    | IFM[1][0], ich[2:3]      |   1    |   11     |   17     | 无，(kh,kw)=(2,2) 无效   | 保持 0                           |    0
//  13-18 | 0  0 |    0    |  0/1   |   7..5  | 无                       |   0    |    -     | 7..16    | 无效坐标                 | 保持 0                           |    0
//  19    | 0  0 |    0    |    0    |    4    | 无                       |   0    |    0     |    4     | IFM[0][0], ich[0:1]      | 首个有效 fold: acc = MAC(fi=0)  |    0
//  20    | 0  0 |    0    |    1    |    4    | 无                       |   0    |    1     |   13     | IFM[0][0], ich[2:3]      | acc += MAC(fi=1)，累加完成       |    0
//  21-26 | 0  0 |    0    |  0/1   |   3..1  | 无                       |   0    |    -     | 3..12    | 无效坐标                 | 保持最终值                       |    0
//  27    | 0  0 |    0    |    0    |    0    | 无                       |   0    |    -     |    0     | 无效坐标                 | 保持最终值                       |    0
//  28    | 0  0 |    0    |    1    |    0    | 无                       |   0    |    -     |    9     | 无效坐标                 | out.write(OFM[0][0][fo=0])       |    1
// -------|-------|---------|---------|---------|--------------------------|--------|----------|----------|--------------------------|----------------------------------|----------
//  29-46 | 0  0 |    1    |  0/1   |   8..0  | 无；fo=1 不触发 prefetch |   0    |    -     | 18..27   | 同样仅 kk=4 有效         | 计算 OFM[0][0][fo=1]             | cycle 46 为 1
// -------|-------|---------|---------|---------|--------------------------|--------|----------|----------|--------------------------|----------------------------------|----------
// 后续    | 0  1 |    0    |  0/1   |    8    | IFM[1][1] 的两个 fold     |   1    | 12,13    | 8,17     | 当前 (oh,ow) 的合法项     | 正常累加                          | 每个 fo 的 kk=0,fi=1 为 1
// ...    | 0  4 |    0    |  0/1   |    8    | IFM[1][4] 的两个 fold     |   1    | 18,19    | 8,17     | 写完整 IFM[1] 后停止预取  | ih_to_read=2                      | 按上述末项有效
```

表中 `RAM_ADDR=10,11` 对应 `line_buf[1][0][0/1]`。由于 `LB_H=3`，第 0、1、2 个
物理行随后循环复用。新调度在每个 `ow` 的首个 kernel 事务中尝试写入一个输入像素；本例中
`oh=0` 的 `ow=0..4` 依次写完 IFM 第 1 行的 5 个列位置。该行写满后 `ih_to_read` 从
1 增加到 2，剩余 `ow` 不再预取，直到下一输出行需要第 2 行。