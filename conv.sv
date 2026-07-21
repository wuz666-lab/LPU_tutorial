// cycle  | cntr_fo | cntr_fi | cntr_kk | in_data      | RAM_WE | RAM_ADDR | MAC输入      | acc状态         | out_valid
// -------|---------|---------|---------|--------------|--------|----------|--------------|----------------|----------
//   1    |    0    |    0    |    0    | ich[0:3],kk0 |   1    |    0     | 0            | x                 |  0     （event:收到一个输入 valid && ready = 1握手成功）
//   2    |    0    |    0    |    1    | ich[0:3],kk1 |   1    |    1     | ich[0:3],kk0 | x   clr 0->1      |  0     （event:收到一个输入 valid && ready = 1握手成功）
//   3    |    0    |    0    |    2    | ich[0:3],kk2 |   1    |    2     | ich[0:3],kk1 | 直接复制kk0        |  0    （event:收到一个输入 valid && ready = 1握手成功）
//   4    |    0    |    0    |    3    | ich[0:3],kk3 |   1    |    3     | ich[0:3],kk2 | R累加 kk1          |  0     （event:收到一个输入 valid && ready = 1握手成功）
//   5    |    0    |    0    |    4    | ich[0:3],kk4 |   1    |    4     | ich[0:3],kk3 | R累加 kk2          |  0 （event:收到一个输入 valid && ready = 1握手成功）
//   6    |    0    |    0    |    5    | ich[0:3],kk5 |   1    |    5     | ich[0:3],kk4 | R累加 kk3          |  0
//   7    |    0    |    0    |    6    | ich[0:3],kk6 |   1    |    6     | ich[0:3],kk5 | R累加 kk4          |  0
//   8    |    0    |    0    |    7    | ich[0:3],kk7 |   1    |    7     | ich[0:3],kk6 | R累加 kk5          |  0
//   9    |    0    |    0    |    8    | ich[0:3],kk8 |   1    |    8     | ich[0:3],kk7 | R累加 kk6          |  0
// -------|---------|---------|---------|--------------|--------|----------|--------------|----------------|----------
//  10    |    0    |    1    |    0    | ich[4:7],kk0 |   1    |    9     | ich[0:3],kk8 | R累加 kk7       |     0
//  11    |    0    |    1    |    1    | ich[4:7],kk1 |   1    |   10     | ich[4:7],kk0 | R累加 kk8       |     0
//  ...   |   ...   |   ...   |   ...   |    ...       |  ...   |   ...    |    ...       | R累加           |     0
//  18    |    0    |    1    |    8    | ich[4:7],kk8 |   1    |   17     | ich[4:7],kk7 | R累加           |     0
// -------|---------|---------|---------|--------------|--------|----------|--------------|----------------|----------
//  19    |    0    |    2    |    0    | ich[8:11],kk0|   1    |   18     | ich[4:7],kk8 | R累加           |    0
//  ...   |   ...   |   ...   |   ...   |    ...       |  ...   |   ...    |    ...       | R累加           |    0
//  27    |    0    |    2    |    8    | ich[8:11],kk8|   1    |   26     |ich[8:11],kk7 | R累加           |    0
// -------|---------|---------|---------|--------------|--------|----------|--------------|----------------|----------
//  28    |    0    |    3    |    0    |ich[12:15],kk0|   1    |   27     |ich[8:11],kk8 | R累加           |    0
//  ...   |   ...   |   ...   |   ...   |    ...       |  ...   |   ...    |    ...       | R累加           |    0
//  36    |    3   |    3    |    8    |ich[12:15],kk8|   1    |   35     |ich[12:15],kk7| R累加             |   0 is_lst_fi_kk=1
// =======|=========|=========|=========|==============|========|==========|==============|================|==========
//  37    |    0   |    0    |    0    |   无新数据    |   0    |    0     |ich[12:15],kk8| R累加          |    0  is_lst_fi_kk_d1=1
//  38    |    0   |    0    |    1    |   无新数据    |   0    |    1     | RAM[0]读出   | R累加 最终结果输出（clr 0->1）|    1 is_lst_fi_kk_d2=1
//  39    |    1    |    0    |    2    |   无新数据    |   0    |    2     | RAM[1]读出   | 直接赋值 kk0    |    0
//  ...   |   ...   |   ...   |   ...   |    ...       |  ...   |   ...    |    ...       | ...           |    ...

// no need to dly 2 cycles

module conv #(
    parameter int unsigned P_ICH      = 4,
    parameter int unsigned P_OCH      = 4,
    parameter int unsigned N_ICH      = 16,
    parameter int unsigned N_OCH      = 16,
    parameter int unsigned K          = 3,
    parameter int unsigned A_BIT      = 8,
    parameter int unsigned W_BIT      = 8,
    parameter int unsigned B_BIT      = 32,
    parameter int unsigned N_HW       = 64,
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

    localparam int unsigned FOLD_I = N_ICH / P_ICH;
    localparam int unsigned FOLD_O = N_OCH / P_OCH;
    localparam int unsigned KK = K * K;
    localparam int unsigned WEIGHT_DEPTH = FOLD_O * FOLD_I * KK;
    localparam int unsigned LB_DEPTH = FOLD_I * KK;
    localparam int unsigned LB_AWIDTH = $clog2(LB_DEPTH);

    logic signed [               B_BIT-1:0] acc                   [  P_OCH];
    logic        [      $clog2(N_HW+1)-1:0] cntr_hw;
    logic        [    $clog2(FOLD_O+1)-1:0] cntr_fo;
    logic        [    $clog2(FOLD_I+1)-1:0] cntr_fi;
    logic        [        $clog2(KK+1)-1:0] cntr_kk;
    logic                                   pipe_en_in;
    logic                                   pipe_en_out;
    logic                                   is_fst_fo;
    logic        [         P_ICH*A_BIT-1:0] in_buf;
    logic                                   is_fst_kk_fi;
    logic                                   is_lst_kk_fi;
    logic                                   line_buffer_we;
    logic        [           LB_AWIDTH-1:0] line_buffer_waddr;
    logic        [         P_ICH*A_BIT-1:0] line_buffer_wdata;
    logic                                   line_buffer_re;
    logic        [           LB_AWIDTH-1:0] line_buffer_raddr;
    logic        [         P_ICH*A_BIT-1:0] line_buffer_rdata;
    logic        [$clog2(WEIGHT_DEPTH)-1:0] weight_addr;
    logic        [   P_OCH*P_ICH*W_BIT-1:0] weight_data;

    rom #(
        .DWIDTH(P_OCH * P_ICH * W_BIT),//kernel number * input channel * weight bit width
        .AWIDTH($clog2(WEIGHT_DEPTH)), 
        .MEM_SIZE(WEIGHT_DEPTH),       //KK * fold In channel * fold Out channel
        .INIT_FILE(W_FILE),
        .ROM_TYPE(W_ROM_TYPE)
    ) u_weight_rom (
        .clk  (clk),
        .ce0  (pipe_en_out),//一直发无效数据，cnt不变，读一个位置
        .addr0(weight_addr),
        .q0   (weight_data)
    );

    ram #(
        .DWIDTH(P_ICH * A_BIT),//channel for input data
        .AWIDTH(LB_AWIDTH),   
        .MEM_SIZE(LB_DEPTH),   //KK * fold number of input channel
        .RAM_STYLE("ultra")
    ) u_line_buffer (
        .clk  (clk),
        .we   (line_buffer_we),
        .waddr(line_buffer_waddr),
        .wdata(line_buffer_wdata),
        .re   (line_buffer_re),
        .raddr(line_buffer_raddr),
        .rdata(line_buffer_rdata)
    );
    assign pipe_en_in           = is_fst_fo ? in_valid : 1'b1; 
    assign pipe_en_out          = out_ready || !out_valid;     //out not block
    assign pipe_en              = pipe_en_in && pipe_en_out;
    assign is_fst_fo            = (cntr_fo == 0);
    assign is_fst_kk_fi         = (cntr_kk == 0) && (cntr_fi == 0) ;  //only when first fold out channel and first fold in channel and first kernel, the input data is valid
    assign is_lst_kk_fi         = (cntr_kk == KK - 1) && (cntr_fi == FOLD_I - 1) && pipe_en_in;  //only when last fold out channel and last fold in channel and last kernel, the input data is valid
    assign in_ready             = is_fst_fo && pipe_en_out;
    assign weight_addr          = (cntr_fo * KK * FOLD_I) + cntr_fi * KK + cntr_kk;
    assign line_buffer_we       = is_fst_fo && in_valid;  //only when first fold out channel, the input data is valid
    assign line_buffer_waddr    = cntr_fi * KK + cntr_kk;
    assign line_buffer_wdata    = in_data;
    assign line_buffer_re       = pipe_en_out;  //一直使能也没事，后面还有判断
    assign line_buffer_raddr    = cntr_fi * KK + cntr_kk;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cntr_hw <= 0;
            cntr_fo <= 0;
            cntr_fi <= 0;
            cntr_kk <= 0;
        end else if (pipe_en) begin
            if (cntr_kk == KK - 1) begin
                cntr_kk <= 0;
                if (cntr_fi == FOLD_I - 1) begin
                    cntr_fi <= 0;
                    if (cntr_fo == FOLD_O - 1) begin
                        cntr_fo <= 0;
                        if (cntr_hw == N_HW - 1) begin
                            cntr_hw <= 0;
                        end else begin
                            cntr_hw <= cntr_hw + 1;
                        end
                    end else begin
                        cntr_fo <= cntr_fo + 1;
                    end
                end else begin
                    cntr_fi <= cntr_fi + 1;
                end
            end else begin
                cntr_kk <= cntr_kk + 1;
            end
        end
    end

    logic mac_array_data_vld_d1;
    logic is_fst_fo_d1;
    logic is_fst_kk_fi_d1;
    logic [P_ICH+1:1] is_lst_kk_fi_dly;
    logic [P_ICH*A_BIT-1:0] in_data_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_data_d1   <= '0;
            is_fst_fo_d1 <= 1'b1;
            is_fst_kk_fi_d1 <= 1'b0;
            is_lst_kk_fi_dly <=  '0;
            mac_array_data_vld_d1 <= 1'b0;
        end else begin
            if (pipe_en_out) begin
                in_data_d1      <= in_data;
                is_fst_fo_d1    <= is_fst_fo;
                is_fst_kk_fi_d1 <= is_fst_kk_fi;
                mac_array_data_vld_d1 <= is_fst_fo ? in_valid : 1'b1;
                is_lst_kk_fi_dly[1] <= is_lst_kk_fi;//1
                for (int i = 1; i < P_ICH+1; i++) begin
                    is_lst_kk_fi_dly[i+1] <= is_lst_kk_fi_dly[i];//1-2 2; 2-3 3;3-4 4; 4-5 5
                end  
            end
        end
        
    end

    assign in_buf = is_fst_fo_d1 ? in_data_d1 : line_buffer_rdata;

    logic        [A_BIT-1:0] x_vec[P_ICH];
    logic signed [W_BIT-1:0] w_vec[P_OCH] [P_ICH];
    always_comb begin
        for (int i = 0; i < P_ICH; i++) begin
            x_vec[i] = in_buf[i*A_BIT+:A_BIT];
        end
    end
    always_comb begin
        for (int o = 0; o < P_OCH; o++) begin
            for (int i = 0; i < P_ICH; i++) begin
                w_vec[o][i] = weight_data[(P_ICH*o+i)*W_BIT+:W_BIT];
            end
        end
    end

    generate
        for (genvar o = 0; o < P_OCH; o++) begin : gen_mac_array
            conv_mac_array #(
                .P_ICH(P_ICH),
                .A_BIT(A_BIT),
                .W_BIT(W_BIT),
                .B_BIT(B_BIT)
            ) u_mac_array (
                .clk    (clk),
                .rst_n  (rst_n),
                .en     (pipe_en_out),
                .dat_vld(mac_array_data_vld_d1),
                .clr    (is_fst_kk_fi_d1),
                .x_vec  (x_vec),
                .w_vec  (w_vec[o]),
                .acc    (acc[o])
            );
        end
    endgenerate

    assign out_valid = is_lst_kk_fi_dly[P_ICH+1];

    always_comb begin
        for (int o = 0; o < P_OCH; o++) begin
            out_data[o*B_BIT+:B_BIT] = acc[o];
        end
    end

endmodule
