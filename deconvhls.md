# Implementation

对于HLS的implementation主要解决以下问题：

1. line buffer ram存储问题应该是计算出来的，计算逻辑如下：

```cpp
/*-------------------------storage depth computation------------------------------------------ */
    // 一个输出行最多同时涉及 ceil(K / S) 行 IFM。
    constexpr unsigned RESIDENT_ROWS = (K - 1) / S + 1;
    // 为边算边存的下一 IFM 行保留一行额外槽，同时避免 N_IH 较小时的资源浪费。
    constexpr unsigned LB_H =
        (RESIDENT_ROWS + 1 < N_IH) ? (RESIDENT_ROWS + 1) : N_IH;
```

2. 对于这整个ofm的运算需要做一个写入ram的初始化：他的目的是写满到oh0所需要的最大的ih行

```cpp
/*--------------------------set up stage ih0------------------------------------------------------ */
    // 启动时写满到 oh=0 可访问的最大 IFM 行，处理 P 很大的情况。
    unsigned ih_to_read = 0;
    unsigned iw_to_read = 0;
    unsigned fi_to_read = 0;
    int max_ih_oh0 = deconv_max_input_coordinate<K, P, S, N_IH>(0);

    while (max_ih_oh0 >= 0 &&
           ih_to_read <= static_cast<unsigned>(max_ih_oh0))
    {
        deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
            in, line_buf, ih_to_read, iw_to_read, fi_to_read);
    }
```

3. 在当前oh的计算期间，进行下一个ofm行oh+1的判断，由于$\frac{(oh+1)-0+P_h}{S}$最多就比$\frac{oh-0+P_h}{S}$大1，毕竟S不可能是小数。那么最多就prefetch再预取1个输入行

```cpp
// 当前 oh 计算期间，若 oh+1 引入新的最高 IFM 行，则预取完整新行。
        unsigned next_ih = (oh + 1 + P) / S;
        bool prefetch_next_input_row =  //flag
            (oh + 1 < N_OH) && (((oh + 1 + P) % S) == 0) &&
            (next_ih < N_IH);
```

4. 预取期间连续写入ram不等待

```cpp
			// 预取期间每个 fi 事务连续写一个输入 beat，直到目标行写满。
      if (prefetch_next_input_row && ih_to_read < N_IH &&
          ih_to_read <= next_ih)
      {
          deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
              in, line_buf, ih_to_read, iw_to_read, fi_to_read);
      }
```

5. MAC输出比新行写入快：在ow行尾出补满预取的新ifm行

```cpp
			// ow 行尾补满正在预取的新 IFM 行；MAC输出比新行ram写入快。
      if (ow == N_OW - 1 && prefetch_next_input_row)
      {
          while (ih_to_read <= next_ih)
          {
              deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
                  in, line_buf, ih_to_read, iw_to_read, fi_to_read);
          }
      }
```

6. MAC计算有效，h_temp/w_temp可以整除S：

```cpp
f (h_temp >= 0 && h_temp % S == 0 && w_temp >= 0 && w_temp % S == 0)
                            {
                                int ih = h_temp / S;
                                int iw = w_temp / S;
                                if (ih >= 0 && ih < N_IH && iw >= 0 && iw < N_IW)
                                {
```

7. 在最后一个fi，最后一个Kernel（kh=0，kh=0）读出

```cpp
 if (kh == 0 && kw == 0 && fi == FOLD_I - 1)
                            {
                                ap_uint<P_OCH * B_BIT> out_buf;
                                for (unsigned poc = 0; poc < P_OCH; ++poc)
                                {
                                    out_buf(SLICE(B_BIT, poc)) = acc[poc];
                                    acc[poc] = 0;
                                }
                                out.write(out_buf);
                            }
```

8. 最后输出未覆盖到的ifm行，消费剩余输入保持协议完整

```cpp
    // 输出边界可能未覆盖所有 IFM 行；消费其余输入以保持流协议完整。
    while (ih_to_read < N_IH)
    {
        deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
            in, line_buf, ih_to_read, iw_to_read, fi_to_read);
    }

    assert(in.empty());
```