module deconv #(
    parameter int unsigned P_ICH      = 4,
    parameter int unsigned P_OCH      = 4,
    parameter int unsigned N_ICH      = 16,
    parameter int unsigned N_OCH      = 16,
    parameter int unsigned N_IH       = 8,
    parameter int unsigned N_IW       = 8,
    parameter int unsigned K          = 3,
    parameter int unsigned P          = 1,
    parameter int unsigned S          = 2,
    parameter int unsigned O_P        = 0,
    parameter int unsigned Z_NUM      = 0,
    parameter int unsigned A_BIT      = 8,
    parameter int unsigned W_BIT      = 8,
    parameter int unsigned B_BIT      = 32,
    parameter string       W_FILE     = "",
    parameter              W_ROM_TYPE = "block"
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [P_ICH*A_BIT-1:0] in_data,
    input  logic                   in_valid,
    output logic                   in_ready,
    output logic [P_OCH*B_BIT-1:0] out_data,
    output logic                   out_valid,
    input  logic                   out_ready
);

    localparam int unsigned N_OH = (N_IH - 1) * S + K - 2 * P + O_P;
    localparam int unsigned N_OW = (N_IW - 1) * S + K - 2 * P + O_P;
    localparam int unsigned FOLD_I = N_ICH / P_ICH;
    localparam int unsigned FOLD_O = N_OCH / P_OCH;
    localparam int unsigned KK = K * K;

    localparam int unsigned WEIGHT_DEPTH = FOLD_O * FOLD_I * KK;
    localparam int unsigned WEIGHT_AWIDTH = (WEIGHT_DEPTH > 1) ? $clog2(WEIGHT_DEPTH) : 1;

    localparam int unsigned RESIDENT_ROWS = (K - 1) / S + 1;
    localparam int unsigned LB_H = ((RESIDENT_ROWS + 1) < N_IH) ?
                                   (RESIDENT_ROWS + 1) : N_IH;

    localparam int unsigned LB_DEPTH = LB_H * N_IW * FOLD_I;
    localparam int unsigned LB_AWIDTH = (LB_DEPTH > 1) ? $clog2(LB_DEPTH) : 1;
    localparam integer P_I = P;

    //ih_max(oh=0)不一定就是P/S，有可能kh=0时P%S并不为0，此时ih_max=P/S无效，需要循环kh
    function automatic integer max_input_coordinate(input integer output_coordinate);
        integer maximum;
        integer kernel_coordinate;
        integer temp;
        integer input_coordinate;
        begin
            maximum = -1;
            for (kernel_coordinate = 0; kernel_coordinate < K;
                 kernel_coordinate = kernel_coordinate + 1) begin
                temp = output_coordinate - kernel_coordinate + P_I;
                if ((temp >= 0) && ((temp % S) == 0)) begin
                    input_coordinate = temp / S;
                    if ((input_coordinate >= 0) && (input_coordinate < N_IH) &&
                        (input_coordinate > maximum)) begin
                        maximum = input_coordinate;
                    end
                end
            end
            max_input_coordinate = maximum;
        end
    endfunction
    //计算oh=0的所需最大输入行
    localparam integer INIT_MAX_IH = max_input_coordinate(0);

    typedef enum logic [1:0] {
        ST_INIT,//输入到oh=0的最大输入行
        ST_PROC,//mac、新行预取
        ST_ROW_WAIT,//如果mac速度大于新行输入速度，等待新行输入完毕
        ST_DRAIN//排空oh=N_OH-1后的剩余输入行
    } state_t;

    state_t state;

    logic [WEIGHT_AWIDTH-1:0] weight_addr;
    logic [P_OCH*P_ICH*W_BIT-1:0] weight_data;

    logic                   line_buffer_we;
    logic [LB_AWIDTH-1:0]   line_buffer_waddr;
    logic [P_ICH*A_BIT-1:0] line_buffer_wdata;
    logic                   line_buffer_re;
    logic [LB_AWIDTH-1:0]   line_buffer_raddr;
    logic [P_ICH*A_BIT-1:0] line_buffer_rdata;

    logic [$clog2(N_OH+1)-1:0] cntr_oh;
    logic [$clog2(N_OW+1)-1:0] cntr_ow;
    logic [$clog2(FOLD_O+1)-1:0] cntr_fo;
    logic [$clog2(K+1)-1:0] cntr_kh;
    logic [$clog2(K+1)-1:0] cntr_kw;
    logic [$clog2(FOLD_I+1)-1:0] cntr_fi;
    //维护三个指针，表示实际的输入坐标推进
    logic [$clog2(N_IH+1)-1:0] ih_to_read;
    logic [$clog2(N_IW+1)-1:0] iw_to_read;
    logic [$clog2(FOLD_I+1)-1:0] fi_to_read;

    logic pipe_en_out;
    logic prefetch_next_input_row;
    logic prefetch_complete;
    logic input_write_completes_prefetch;
    logic drain_needed;
    logic final_output_seen;

    integer next_ih;
    integer h_temp;
    integer w_temp;
    integer ih;
    integer iw;
    logic valid_pos;
    logic is_fst_kh_kw_fi;
    logic is_lst_kh_kw_fi;
    logic is_final_output;

    logic read_valid_d1;
    logic clear_d1;
    logic [P_ICH+1:1] output_valid_dly;
    logic [P_ICH+1:1] final_output_dly;

    logic signed [B_BIT-1:0] acc[P_OCH];
    logic [A_BIT-1:0] x_vec[P_ICH];
    logic signed [W_BIT-1:0] w_vec[P_OCH][P_ICH];
    /*------------------------------comb begin----------------------------------*/
    assign    h_temp = $signed({1'b0, cntr_oh}) - $signed({1'b0, cntr_kh}) + P_I;
    assign    w_temp = $signed({1'b0, cntr_ow}) - $signed({1'b0, cntr_kw}) + P_I;
    assign    ih = h_temp / S;
    assign    iw = w_temp / S;
    //预取输入行
    assign    next_ih = (cntr_oh + 1 + P_I) / S;
    /*-----------------------------comb end-------------------------------------*/
    /*-----------------------------flag begin-----------------------------------*/
    //是否是有效输入像素
    assign    valid_pos = (h_temp >= 0) && ((h_temp % S) == 0) &&
                          (w_temp >= 0) && ((w_temp % S) == 0) &&
                          (ih >= 0) && (ih < N_IH) &&
                          (iw >= 0) && (iw < N_IW);
    //是否预取
    assign    prefetch_next_input_row = (cntr_oh + 1 < N_OH) &&
                                        (((cntr_oh + 1 + P_I) % S) == 0) &&
                                        (next_ih < N_IH);
    //是否预取完成
    assign prefetch_complete = !prefetch_next_input_row || (ih_to_read > next_ih);
    //最后一笔数据输入时拉高
    assign input_write_completes_prefetch = prefetch_next_input_row &&
                                            (ih_to_read == next_ih) &&
                                            (iw_to_read == N_IW - 1) &&
                                            (fi_to_read == FOLD_I - 1);
    //是否需要排空剩余ifm数据
    assign drain_needed = ih_to_read < N_IH;
    /*-----------------------------flag end-------------------------------------*/
    assign is_fst_kh_kw_fi = (cntr_kh == K - 1) &&
                             (cntr_kw == K - 1) && (cntr_fi == 0);
    assign is_lst_kh_kw_fi = (cntr_kh == 0) &&
                             (cntr_kw == 0) && (cntr_fi == FOLD_I - 1);
    assign is_final_output = (cntr_oh == N_OH - 1) &&
                             (cntr_ow == N_OW - 1) &&
                             (cntr_fo == FOLD_O - 1) && is_lst_kh_kw_fi;

    assign pipe_en_in  = in_ready && in_valid;
    assign pipe_en_out = out_ready || !out_valid;

    always_comb begin
        case (state)
            ST_INIT: begin
                in_ready = (INIT_MAX_IH >= 0);
            end
            ST_PROC: begin
                in_ready = prefetch_next_input_row && !prefetch_complete && pipe_en_out;
            end
            ST_ROW_WAIT: begin
                in_ready = !prefetch_complete && pipe_en_out;
            end
            ST_DRAIN: begin
                in_ready = drain_needed && pipe_en_out;
            end
            default: begin
                in_ready = 1'b0;
            end
        endcase
    end

    assign line_buffer_we = pipe_en_in;
    assign line_buffer_waddr = ((ih_to_read % LB_H) * N_IW * FOLD_I) +
                               (iw_to_read * FOLD_I) + fi_to_read;
    assign line_buffer_wdata = in_data;
    assign line_buffer_re = (state == ST_PROC) && pipe_en_out && valid_pos;
    
    always_comb begin
        if (valid_pos) begin
            line_buffer_raddr = ((ih % LB_H) * N_IW * FOLD_I) +
                                (iw * FOLD_I) + cntr_fi;
        end else begin
            line_buffer_raddr = '0;
        end
    end

    assign weight_addr = (cntr_fo * FOLD_I * KK) +
                         (cntr_fi * KK) + (cntr_kh * K) + cntr_kw;

    rom #(
        .DWIDTH(P_OCH * P_ICH * W_BIT),
        .AWIDTH(WEIGHT_AWIDTH),
        .MEM_SIZE(WEIGHT_DEPTH),
        .INIT_FILE(W_FILE),
        .ROM_TYPE(W_ROM_TYPE)
    ) u_weight_rom (
        .clk(clk),
        .ce0((state == ST_PROC) && pipe_en_out),
        .addr0(weight_addr),
        .q0(weight_data)
    );

    ram #(
        .DWIDTH(P_ICH * A_BIT),
        .AWIDTH(LB_AWIDTH),
        .MEM_SIZE(LB_DEPTH)
    ) u_line_buffer (
        .clk(clk),
        .we(line_buffer_we),
        .waddr(line_buffer_waddr),
        .wdata(line_buffer_wdata),
        .re(line_buffer_re),
        .raddr(line_buffer_raddr),
        .rdata(line_buffer_rdata)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_valid_d1 <= 1'b0;
            clear_d1 <= 1'b0;
            output_valid_dly <= '0;
            final_output_dly <= '0;
        end else if (pipe_en_out) begin
            read_valid_d1 <= (state == ST_PROC) && valid_pos;//发射
            clear_d1 <= (state == ST_PROC) && is_fst_kh_kw_fi;
            output_valid_dly[1] <= (state == ST_PROC) && is_lst_kh_kw_fi;
            final_output_dly[1] <= (state == ST_PROC) && is_final_output;
            //卷积计算到了某一行 oh 的末尾（ow = N_OW-1），主控状态机（管线头部）判断需要等待新的一行 IFM 数据写入，于是 state 从 ST_PROC 跳转到了 ST_ROW_WAIT。
            //让前面周期派发的计算接着向后流
            for (int delay_index = 1; delay_index < P_ICH + 1;
                 delay_index = delay_index + 1) begin
                output_valid_dly[delay_index+1] <= output_valid_dly[delay_index];
                final_output_dly[delay_index+1] <= final_output_dly[delay_index];
            end
        end
    end

    // always_ff @(posedge clk or negedge rst_n) begin
    //     if (!rst_n) begin
    //         read_valid_d1 <= 1'b0;
    //         clear_d1 <= 1'b0;
    //         output_valid_dly <= '0;
    //         final_output_dly <= '0;
    //     end else if (pipe_en_out && (state == ST_PROC)) begin
    //         read_valid_d1 <= valid_pos;
    //         clear_d1 <= is_fst_kh_kw_fi;
    //         output_valid_dly[1] <= is_lst_kh_kw_fi;
    //         final_output_dly[1] <= is_final_output;
    //         for (int delay_index = 1; delay_index < P_ICH + 1;
    //              delay_index = delay_index + 1) begin
    //             output_valid_dly[delay_index+1] <= output_valid_dly[delay_index];
    //             final_output_dly[delay_index+1] <= final_output_dly[delay_index];
    //         end
    //     end
    // end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_INIT;
            cntr_oh <= '0;
            cntr_ow <= '0;
            cntr_fo <= '0;
            cntr_kh <= K - 1;
            cntr_kw <= K - 1;
            cntr_fi <= '0;
            ih_to_read <= '0;
            iw_to_read <= '0;
            fi_to_read <= '0;
            final_output_seen <= 1'b0;
        end else begin
            if (final_output_dly[P_ICH+1] && out_ready) begin
                final_output_seen <= 1'b1;
            end

            if (line_buffer_we) begin
                if (fi_to_read == FOLD_I - 1) begin
                    fi_to_read <= '0;
                    if (iw_to_read == N_IW - 1) begin
                        iw_to_read <= '0;
                        ih_to_read <= ih_to_read + 1'b1;
                    end else begin
                        iw_to_read <= iw_to_read + 1'b1;
                    end
                end else begin
                    fi_to_read <= fi_to_read + 1'b1;
                end
            end

            case (state)
                ST_INIT: begin
                    if (INIT_MAX_IH < 0) begin
                        state <= ST_PROC;
                        final_output_seen <= 1'b0;
                    end else if (line_buffer_we && (ih_to_read == INIT_MAX_IH) &&
                                 (iw_to_read == N_IW - 1) &&
                                 (fi_to_read == FOLD_I - 1)) begin
                        state <= ST_PROC;
                        final_output_seen <= 1'b0;
                    end
                end

                ST_PROC: begin
                    if (pipe_en_out) begin
                        if (is_lst_kh_kw_fi) begin
                            if (cntr_fo == FOLD_O - 1) begin
                                cntr_fo <= '0;
                                if (cntr_ow == N_OW - 1) begin
                                    cntr_ow <= '0;
                                    if (cntr_oh == N_OH - 1) begin
                                        state <= ST_DRAIN;
                                    end else if (prefetch_next_input_row) begin
                                        state <= ST_ROW_WAIT;
                                    end else begin
                                        cntr_oh <= cntr_oh + 1'b1;
                                    end
                                end else begin
                                    cntr_ow <= cntr_ow + 1'b1;
                                end
                            end else begin
                                cntr_fo <= cntr_fo + 1'b1;
                            end
                        end

                        if (cntr_fi == FOLD_I - 1) begin
                            cntr_fi <= '0;
                            if (cntr_kw == 0) begin
                                cntr_kw <= K - 1;
                                if (cntr_kh == 0) begin
                                    cntr_kh <= K - 1;
                                end else begin
                                    cntr_kh <= cntr_kh - 1'b1;
                                end
                            end else begin
                                cntr_kw <= cntr_kw - 1'b1;
                            end
                        end else begin
                            cntr_fi <= cntr_fi + 1'b1;
                        end
                    end
                end

                ST_ROW_WAIT: begin
                    if (pipe_en_out && (prefetch_complete ||
                                        (in_valid && input_write_completes_prefetch))) begin
                        state <= ST_PROC;
                        cntr_oh <= cntr_oh + 1'b1;
                    end
                end

                ST_DRAIN: begin
                    if (!drain_needed && final_output_seen) begin
                        state <= ST_INIT;
                        cntr_oh <= '0;
                        cntr_ow <= '0;
                        cntr_fo <= '0;
                        cntr_kh <= K - 1;
                        cntr_kw <= K - 1;
                        cntr_fi <= '0;
                        ih_to_read <= '0;
                        iw_to_read <= '0;
                        fi_to_read <= '0;
                        final_output_seen <= 1'b0;
                    end
                end

                default: begin
                    state <= ST_INIT;
                end
            endcase
        end
    end

    always_comb begin
        for (int input_index = 0; input_index < P_ICH; input_index++) begin
            x_vec[input_index] = read_valid_d1 ?
                                 line_buffer_rdata[input_index*A_BIT+:A_BIT] : '0;
        end
        for (int output_index = 0; output_index < P_OCH; output_index++) begin
            for (int input_index = 0; input_index < P_ICH; input_index++) begin
                w_vec[output_index][input_index] =
                    weight_data[(P_ICH*output_index+input_index)*W_BIT+:W_BIT];
            end
        end
    end

    generate
        for (genvar output_index = 0; output_index < P_OCH;
             output_index = output_index + 1) begin : gen_mac_array
            conv_mac_array #(
                .P_ICH(P_ICH),
                .A_BIT(A_BIT),
                .W_BIT(W_BIT),
                .B_BIT(B_BIT)
            ) u_mac_array (
                .clk(clk),
                .rst_n(rst_n),
                .en(pipe_en_out),
                .dat_vld(read_valid_d1),
                .clr(clear_d1),
                .x_vec(x_vec),
                .w_vec(w_vec[output_index]),
                .acc(acc[output_index])
            );
        end
    endgenerate

    assign out_valid = output_valid_dly[P_ICH+1];

    always_comb begin
        for (int output_index = 0; output_index < P_OCH; output_index++) begin
            out_data[output_index*B_BIT+:B_BIT] = acc[output_index];
        end
    end

endmodule