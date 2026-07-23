# Deconvolution RTL Implementation

本文档对应 `deconvhls.md` 中的 HLS 逻辑，记录了 `deconv.sv` 和 `deconv_mac.sv` 在 RTL 级别的实现思路和映射关系。

## 1. Line Buffer 存储深度配置

与 HLS 保持一致，计算最大同时常驻的输入行数 `RESIDENT_ROWS`，并为边算边存的下一输入行预留一行槽位。

```verilog
    localparam int unsigned RESIDENT_ROWS = (K - 1) / S + 1;
    localparam int unsigned LB_H = ((RESIDENT_ROWS + 1) < N_IH) ?
                                   (RESIDENT_ROWS + 1) : N_IH;
    localparam int unsigned LB_DEPTH = LB_H * N_IW * FOLD_I;
```

## 2. 初始数据填充 (ST_INIT)

RTL 在 `ST_INIT` 状态下完成启动前的数据预读，写满到 `oh=0` 所需的最大行 `INIT_MAX_IH`，这完全对应于 HLS 的 `deconv_max_input_coordinate` 函数逻辑。

```verilog
    function automatic integer max_input_coordinate(input integer output_coordinate);
        // ... 计算 oh=0 时涉及的最大输入坐标 ih ...
    endfunction
    localparam integer INIT_MAX_IH = max_input_coordinate(0);

// 状态流转条件：当写满 INIT_MAX_IH 行时，跳转至 ST_PROC
    ST_INIT: begin
        if (INIT_MAX_IH < 0) begin
            state <= ST_PROC;
        end else if (line_buffer_we && (ih_to_read == INIT_MAX_IH) &&
                     (iw_to_read == N_IW - 1) && (fi_to_read == FOLD_I - 1)) begin
            state <= ST_PROC;
        end
    end
```

## 3. 逐行计算与预取判定 (ST_PROC & ST_ROW_WAIT)

状态机在计算当前的 `oh` 层级时，使用 `next_ih` 和 `prefetch_next_input_row` 实时判断下一层级是否可以提前加载 IFM 行。

```verilog
        next_ih = (cntr_oh + 1 + P_I) / S;
        prefetch_next_input_row = (cntr_oh + 1 < N_OH) &&
                                  (((cntr_oh + 1 + P_I) % S) == 0) &&
                                  (next_ih < N_IH);
        prefetch_complete = !prefetch_next_input_row || (ih_to_read > next_ih);
```

若当前行 `oh` 的 MAC 已经调度计算到最后，但新一行的预取还未完全写满，则状态机会离开 `ST_PROC` 进入 `ST_ROW_WAIT` 等待补齐：

```verilog
// 在 ST_PROC 计算到 ow = N_OW - 1 尾部时的状态判断：
if (cntr_oh == N_OH - 1) begin
    state <= ST_DRAIN;
end else if (prefetch_next_input_row) begin
    // 若此时需要新行（等待写满），转入 ROW_WAIT；
    state <= ST_ROW_WAIT;
end else begin
    cntr_oh <= cntr_oh + 1'b1;
end
```

## 4. 边界处理与无效坐标过滤

MAC 运算时可能出现坐标超越物理维度（因 PADDING 和步长错位），对应 HLS 的合法性 `if` 检查。此时使用 `valid_pos` 标识。若为无效坐标，会阻断 `line_buffer_re`，同时输入喂 `0`，避免未初始化的 X 态污染流水线累加结果。

```verilog
        valid_pos = (h_temp >= 0) && ((h_temp % S) == 0) &&
                    (w_temp >= 0) && ((w_temp % S) == 0) &&
                    (ih >= 0) && (ih < N_IH) &&
                    (iw >= 0) && (iw < N_IW);

// x_vec 向量在无效坐标下强制为 0
        x_vec[input_index] = read_valid_d1 ?
                             line_buffer_rdata[input_index*A_BIT+:A_BIT] : '0;
```

## 5. 计算流水与结果输出判定

判定前计算流是否处于最后一组卷积折叠步（`kh=0`, `kw=0`, `fi=FOLD_I-1`），进而决定是否推出 MAC 阵列的结果。该输出伴随 `valid_dly` 并行流水延迟多拍最终拉高 `out_valid`。

```verilog
        is_lst_kh_kw_fi = (cntr_kh == 0) &&
                           (cntr_kw == 0) && (cntr_fi == FOLD_I - 1);
        
        // 跟随延迟通过 pipeline 传递
        output_valid_dly[1] <= calc_issue && is_lst_kh_kw_fi;
        assign out_valid = output_valid_dly[P_ICH+1];
```

## 6. 末尾状态排空协议 (ST_DRAIN)

为了应对大 PADDING 等导致计算遍历早于输入数据写完的边界情况。利用 `ST_DRAIN` 汲取遗留流信号数据，使其符合 AXI-Stream 等总线的非阻塞协议规范（消耗不需要的数据），与 HLS `while(ih_to_read < N_IH)` 等效。

```verilog
        drain_needed = ih_to_read < N_IH;
        
        ST_DRAIN: begin
            in_ready = drain_needed && pipe_en_out;
            if (!drain_needed && final_output_seen) begin
                state <= ST_INIT; 
            end
        end
```

## 7. MAC 分布式计算阵列 (`deconv_mac.sv`)

相比原 HLS 的大宽幅 for-loop 加和，在 RTL 中采用了串联级联结构的阵列计算，通过依次传递和累加规避过高扇入逻辑：

```verilog
    generate
        for (genvar i = 0; i < P_ICH - 1; i++) begin : gen_mac
            logic signed [B_BIT-1:0] acc_r;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    acc_r <= '0;
                end else if (en) begin
                    if (dat_vld) begin
                        acc_r <= ($signed(x_vec[i]) * w_vec[i]) - mac_cascade[i];
                    end else if (clr) begin
                        acc_r <= '0;
                    end
                end
            end
            assign mac_cascade[i+1] = acc_r;
        end
    endgenerate
```

并在 `mac_tail` 部分执行最后一个 Channel 通道的组装，得到本拍的并联部分和，在时序上完成了 `(S+1)` 行缓冲下所需的数据通量消解。
