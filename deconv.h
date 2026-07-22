// Line buffer 仅保留一个输出行会访问的 IFM 行，避免存储完整 IFM。
// 启动时预读到 oh=0 实际访问的最大 ih；常规配置沿用像素级预取。
// 当大 padding 使 OFM 宽度不足以完成一整行预取时，退化为逐输出行补齐。

#pragma once
#ifndef CONV_H_
#define CONV_H_

#include "stream_tools.h"

template<
    unsigned N_ICH,
    unsigned N_OCH,
    unsigned N_IH,
    unsigned N_IW,
    unsigned K,
    unsigned P,
    unsigned S,
    unsigned O_P,
    unsigned BIT_ACTV,
    unsigned BIT_WGHT,
    unsigned BIT_CONV
>
void deconv_golden(
    data_stream<BIT_ACTV>& in,
    data_stream<BIT_CONV>& out,
    const ap_int<BIT_WGHT> weight[N_OCH][K*K][N_ICH]
)
{
    constexpr unsigned N_OH = (N_IH - 1) * S + K - 2 * P + O_P;
    constexpr unsigned N_OW = (N_IW - 1) * S + K - 2 * P + O_P;
    ap_int<BIT_ACTV> line_buf[N_IH][N_IW][N_ICH];
    ap_int<BIT_CONV> output_buf[N_OH][N_OW][N_OCH];

    for(unsigned ih = 0; ih < N_IH; ++ih) {
        for(unsigned iw = 0; iw < N_IW; ++iw) {
            for(unsigned fi = 0; fi < N_ICH; ++fi) {
                line_buf[ih][iw][fi] = in.read();
            }
        }
    }

    for(unsigned oh = 0; oh < N_OH; ++oh) {
        for(unsigned ow = 0; ow < N_OW; ++ow) {
            for(unsigned oc = 0; oc < N_OCH; ++oc) {
                ap_int<BIT_CONV> acc = 0;
                for(unsigned kh = 0; kh < K; ++kh) {
                    for(unsigned kw = 0; kw < K; ++kw) {
                        int h_temp = oh - kh + P;
                        int w_temp = ow - kw + P;
                        if(h_temp >= 0 && h_temp % S == 0 && w_temp >= 0 && w_temp % S == 0) {
                            int ih = h_temp / S;
                            int iw = w_temp / S;
                            if(ih >= 0 && ih < N_IH && iw >= 0 && iw < N_IW) {
                                for(unsigned fi = 0; fi < N_ICH; ++fi) {
                                    ap_int<BIT_ACTV> x = line_buf[ih][iw][fi];
                                    ap_int<BIT_WGHT> w = weight[oc][kh * K + kw][fi];
                                    acc += x * w;
                                }
                            }
                        }
                    }
                }
                out.write(acc);
            }
        }
    }

    assert(in.empty());
    assert(out.size() == N_OH * N_OW * N_OCH);
}
//计算ow/oh需要的最大ifm的iw/ih
template <unsigned K,
          unsigned P,
          unsigned S,
          unsigned N_I>
int deconv_max_input_coordinate(unsigned output_coordinate)
{
    int maximum = -1;
    for (unsigned kernel_coordinate = 0; kernel_coordinate < K; ++kernel_coordinate)
    {
        int temp = static_cast<int>(output_coordinate) -
                   static_cast<int>(kernel_coordinate) + static_cast<int>(P);
        if (temp >= 0 && temp % S == 0)
        {
            int input_coordinate = temp / S;
            if (input_coordinate >= 0 && input_coordinate < static_cast<int>(N_I) &&
                input_coordinate > maximum)
            {
                maximum = input_coordinate;
            }
        }
    }
    return maximum;
}
//维护三个全局指针
//write ram,每次调用只写入一个输入beat，推进三个指针
template <unsigned P_ICH,
          unsigned A_BIT,
          unsigned LB_H,
          unsigned N_IW,
          unsigned FOLD_I>
void deconv_write_input_beat(
    data_stream<P_ICH * A_BIT>& in,
    ap_uint<P_ICH * A_BIT> line_buf[LB_H][N_IW][FOLD_I],
    unsigned& ih_to_read,
    unsigned& iw_to_read,
    unsigned& fi_to_read)
{
    line_buf[ih_to_read % LB_H][iw_to_read][fi_to_read] = in.read();
    ++fi_to_read;
    if (fi_to_read == FOLD_I)
    {
        fi_to_read = 0;
        ++iw_to_read;
        if (iw_to_read == N_IW)
        {
            iw_to_read = 0;
            ++ih_to_read;
        }
    }
}

template <unsigned P_ICH,
          unsigned P_OCH,
          unsigned N_ICH,
          unsigned N_OCH,
          unsigned N_IH,
          unsigned N_IW,
          unsigned K,
          unsigned P,
          unsigned S,
          unsigned O_P,
          unsigned A_BIT,
          unsigned W_BIT,
          unsigned B_BIT>
void deconv(data_stream<P_ICH * A_BIT>& in,
               data_stream<P_OCH * B_BIT>& out,
               const ap_uint<P_OCH * P_ICH * W_BIT> weight[N_OCH / P_OCH][N_ICH / P_ICH][K * K])
{
    constexpr unsigned N_OH = (N_IH - 1) * S + K - 2 * P + O_P;
    constexpr unsigned N_OW = (N_IW - 1) * S + K - 2 * P + O_P;
    constexpr unsigned FOLD_I = N_ICH / P_ICH;
    constexpr unsigned FOLD_O = N_OCH / P_OCH;
/*-------------------------storage depth computation------------------------------------------ */
    // 一个输出行最多同时涉及 ceil(K / S) 行 IFM。
    constexpr unsigned RESIDENT_ROWS = (K - 1) / S + 1;
    // 为边算边存的下一 IFM 行保留一行额外槽，同时避免 N_IH 较小时的资源浪费。
    constexpr unsigned LB_H =
        (RESIDENT_ROWS + 1 < N_IH) ? (RESIDENT_ROWS + 1) : N_IH;
/*-------------------------end-----------------------------------------------------------------*/
    ap_uint<P_ICH * A_BIT> line_buf[LB_H][N_IW][FOLD_I];
    ap_int<B_BIT> acc[P_OCH];
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
/*------------------------end----------------------------------------------------------------- */
    for (unsigned oh = 0; oh < N_OH; ++oh)
    {
        // 当前 oh 计算期间，若 oh+1 引入新的最高 IFM 行，则预取完整新行。
        unsigned next_ih = (oh + 1 + P) / S;
        bool prefetch_next_input_row =  //flag
            (oh + 1 < N_OH) && (((oh + 1 + P) % S) == 0) &&
            (next_ih < N_IH);

        for (unsigned ow = 0; ow < N_OW; ++ow)
        {
            for (unsigned fo = 0; fo < FOLD_O; ++fo)
            {
                // ---- 清零累加器 ----
                for (unsigned poc = 0; poc < P_OCH; ++poc)
                {
                    acc[poc] = 0;
                }
                // ---- 卷积核循环 (倒序, 与正序数学等价) ----
                for (signed kh = K - 1; kh >= 0; kh--)
                {
                    for (signed kw = K - 1; kw >= 0; kw--)
                    {
                        int h_temp = oh - kh + P;
                        int w_temp = ow - kw + P;
                        for (unsigned fi = 0; fi < FOLD_I; ++fi)
                        {
                            ap_uint<P_OCH * P_ICH * W_BIT> wt_buf = weight[fo][fi][kh * K + kw];
                            // 预取期间每个 fi 事务连续写一个输入 beat，直到目标行写满。
                            if (prefetch_next_input_row && ih_to_read < N_IH &&
                                ih_to_read <= next_ih)
                            {
                                deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
                                    in, line_buf, ih_to_read, iw_to_read, fi_to_read);
                            }

                            // ---- MAC 计算 ----
                            if (h_temp >= 0 && h_temp % S == 0 && w_temp >= 0 && w_temp % S == 0)
                            {
                                int ih = h_temp / S;
                                int iw = w_temp / S;
                                if (ih >= 0 && ih < N_IH && iw >= 0 && iw < N_IW)
                                {
                                    for (unsigned pic = 0; pic < P_ICH; ++pic)
                                    {
                                        unsigned ic = fi * P_ICH + pic;
                                        ap_uint<P_ICH * A_BIT> in_buf = line_buf[ih % LB_H][iw][fi];
                                        ap_uint<A_BIT> x = in_buf(SLICE(A_BIT, pic));
                                        for (unsigned poc = 0; poc < P_OCH; ++poc)
                                        {
                                            unsigned oc = fo * P_OCH + poc;
                                            ap_int<W_BIT> w =
                                                wt_buf(SLICE(W_BIT, P_ICH * poc + pic));
                                            acc[poc] += x * w;
                                        }
                                    }
                                }
                            }

                            // ---- 输出: 仅在最后一个 fi、最后一个 kernel 位置时写出 ----
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
                        }  // fi
                    }  // kw
                }  // kh
            }  // fo
            // ow 行尾补满正在预取的新 IFM 行；MAC输出比新行ram写入快。
            if (ow == N_OW - 1 && prefetch_next_input_row)
            {
                while (ih_to_read <= next_ih)
                {
                    deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
                        in, line_buf, ih_to_read, iw_to_read, fi_to_read);
                }
            }
        }  // ow
    }  // oh

    // 输出边界可能未覆盖所有 IFM 行；消费其余输入以保持流协议完整。
    while (ih_to_read < N_IH)
    {
        deconv_write_input_beat<P_ICH, A_BIT, LB_H, N_IW, FOLD_I>(
            in, line_buf, ih_to_read, iw_to_read, fi_to_read);
    }

    assert(in.empty());
    // assert(out.size() == N_OH * N_OW * FOLD_O);
}


#endif


// // ---- 卷积核循环 (倒序, 与正序数学等价) ----
//                 for (signed kh = K - 1; kh >= 0; kh--)
//                 {
//                     for (signed kw = K - 1; kw >= 0; kw--)
//                     {
//                         int h_temp = oh - kh + P;
//                         int w_temp = ow - kw + P;

//                         for (unsigned fi = 0; fi < FOLD_I; ++fi)
//                         {
//                             ap_uint<P_OCH * P_ICH * W_BIT> wt_buf = weight[fo][fi][kh * K + kw];

//                             // ---- Prefetch: 条件成立时读入下一行 IFM 数据 ----
//                             // 条件: 输出坐标映射到有效输入坐标 + 本轮第一个 kernel 元素 + 第一个输出通道折叠
//                             if (oh % S == 0 && ow % S == 0 && kh == K - 1 && kw == K - 1 && fo == 0)
//                             {
//                                 unsigned iw = ow / S;
//                                 // BUGFIX: 检查 iw 不越界
//                                 if (iw < N_IW && ih_to_read < N_IH)
//                                 {
//                                     line_buf[ih_to_read % LB_H][iw][fi] = in.read();
//                                     if (iw == N_IW - 1 && fi == FOLD_I - 1)
//                                     {
//                                         ih_to_read++;
//                                     }
//                                 }
//                             }

//                             // ---- MAC 计算 ----
//                             if (h_temp >= 0 && h_temp % S == 0 && w_temp >= 0 && w_temp % S == 0)
//                             {
//                                 int ih = h_temp / S;
//                                 int iw = w_temp / S;
//                                 if (ih >= 0 && ih < N_IH && iw >= 0 && iw < N_IW)
//                                 {
//                                     for (unsigned pic = 0; pic < P_ICH; ++pic)
//                                     {
//                                         unsigned ic = fi * P_ICH + pic;
//                                         ap_uint<P_ICH * A_BIT> in_buf = line_buf[ih % LB_H][iw][fi];
//                                         ap_uint<A_BIT> x = in_buf(SLICE(A_BIT, pic));
//                                         for (unsigned poc = 0; poc < P_OCH; ++poc)
//                                         {
//                                             unsigned oc = fo * P_OCH + poc;
//                                             ap_int<W_BIT> w =
//                                                 wt_buf(SLICE(W_BIT, P_ICH * poc + pic));
//                                             acc[poc] += x * w;
//                                         }
//                                     }
//                                 }
//                             }

//                             // ---- 输出: 仅在最后一个 fi、最后一个 kernel 位置时写出 ----
//                             // BUGFIX: 原来缺少 fi==FOLD_I-1 条件, 导致每个 fi 都输出一次 (FOLD_I 次)
//                             if (kh == 0 && kw == 0 && fi == FOLD_I - 1)
//                             {
//                                 ap_uint<P_OCH * B_BIT> out_buf;
//                                 for (unsigned poc = 0; poc < P_OCH; ++poc)
//                                 {
//                                     out_buf(SLICE(B_BIT, poc)) = acc[poc];
//                                     acc[poc] = 0;
//                                 }
//                                 out.write(out_buf);
//                             }
//                         }  // fi
//                     }  // kw
//                 }  // kh
//             }  // fo
//         }  // ow
//     }  // oh