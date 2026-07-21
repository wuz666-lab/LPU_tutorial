// | cycle | row,col | w[0] | pixel_buf | temp_max | lb_we | lb_re | RAM输出     | h_max_reg  | out_valid |
// | ----: | ------- | ---- | --------- | -------- | ----- | ----- | --------- | ---------- | --------- |
// |     0 | (0,0)   | 0    | 写A        | 无        | 0     | 0     | -         | -          | 0         |
// |     1 | (0,1)   | 1    | -         | max(A,B) | 1     | 0     | -         | -          | 0         |
// |     2 | (1,0)   | 0    | 写C        | 无        | 0     | 0     | -         | -          | 0         |
// |     3 | (1,1)   | 1    | -         | max(C,D) | 0     | 1     | 下一拍输出     | 写入max(C,D) | 0         |
// |     4 | (2,0)   | 0    | 写E        | 无        | 0     | 0     | row0水平max | 保持         | 1 max(max(A,B),max(C,D))        |
// |     5 | (2,1)   | 1    | -         | max(E,F) | 1     | 0     | -         | -          | 0         |
// |     6 | (3,0)   | 0    | 写G        | 无        | 0     | 0     | -         | -          | 0         |
// |     7 | (3,1)   | 1    | -         | max(G,H) | 0     | 1     | 下一拍输出     | 写入max(G,H) | 0         |
// |     8 | (4,0)   | 0    | ...       | ...      | ...   | ...   | row2水平max | ...        | 1  max(max(E,F),max(G,H))        |



module maxpool_2x2 #(
    parameter int unsigned P_CH  = 4,
    parameter int unsigned N_CH  = 16,
    parameter int unsigned N_IW  = 64,
    parameter int unsigned A_BIT = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [P_CH*A_BIT-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,
    output logic [P_CH*A_BIT-1:0] out_data,
    output logic                  out_valid,
    input  logic                  out_ready
);

    localparam int unsigned FOLD = N_CH / P_CH;
    localparam int unsigned N_OW = N_IW / 2;
    localparam int unsigned LB_DEPTH = N_OW * FOLD;
    localparam int unsigned LB_AWIDTH = $clog2(LB_DEPTH);

    logic                      cntr_h;
    logic [$clog2(N_IW+1)-1:0] cntr_w;
    logic [$clog2(FOLD+1)-1:0] cntr_f;
    logic                      lb_we;
    logic [     LB_AWIDTH-1:0] lb_waddr;
    logic [    P_CH*A_BIT-1:0] lb_wdata;
    logic                      lb_re;
    logic [     LB_AWIDTH-1:0] lb_raddr;
    logic [    P_CH*A_BIT-1:0] lb_rdata;
    logic [    P_CH*A_BIT-1:0] pixel_buf        [FOLD];
    logic                      pipe_en;
    logic [    P_CH*A_BIT-1:0] temp_max_data;
    //counter: f-w-h
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cntr_h <= 1;
            cntr_w <= 0;
            cntr_f <= 0;
        end else begin
            if (pipe_en) begin
                if (cntr_f == FOLD - 1) begin
                    cntr_f <= 0;
                    if (cntr_w == N_IW - 1) begin
                        cntr_w <= 0;
                        cntr_h <= cntr_h + 1;
                    end else begin
                        cntr_w <= cntr_w + 1;
                    end
                end else begin
                    cntr_f <= cntr_f + 1;
                end
            end
        end
    end

    assign in_ready     = out_ready || !out_valid;  //输出背压了
    assign pipe_en      = in_valid && in_ready;     //上游有数据且输出没被背压

    //indata -> pixel_buf[cntr_f] -> temp_max_data -> lb_wdata
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int unsigned i = 0; i < FOLD; i++) begin
                pixel_buf[i] <= '1;
            end
        end else if (pipe_en && cntr_w[0] == 1'b0) begin
            pixel_buf[cntr_f] <= in_data;
        end
    end

    assign temp_max_data    = max_vec(pixel_buf[cntr_f], in_data);
    assign lb_wdata         = temp_max_data;
    assign lb_we            = pipe_en && (cntr_w[0] == 1'b1) && (cntr_h == 1'b1);
    assign lb_waddr         = (cntr_w >> 1) * FOLD + cntr_f;
    assign lb_re            = pipe_en && (cntr_w[0] == 1'b1) && (cntr_h == 1'b0);//cycle4 read address
    assign lb_raddr         = (cntr_w >> 1) * FOLD + cntr_f;//cycle4 read address

    ram #(
        .DWIDTH  (P_CH * A_BIT),
        .AWIDTH  (LB_AWIDTH),
        .MEM_SIZE(LB_DEPTH)
    ) u_line_buf (
        .clk  (clk),
        .we   (lb_we),
        .waddr(lb_waddr),
        .wdata(lb_wdata),
        .re   (lb_re),
        .raddr(lb_raddr),
        .rdata(lb_rdata)
    );

    logic [P_CH*A_BIT-1:0]     h_max_reg [FOLD];   
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int unsigned i = 0; i < FOLD; i++) h_max_reg[i] <= '0;
        end else if (pipe_en && (cntr_w[0] == 1'b1) && (cntr_h == 1'b0)) begin
            h_max_reg[cntr_f] <= temp_max_data;
        end
    end

    
    logic lb_re_d;//打拍输出，解耦合
    logic [$clog2(FOLD+1)-1:0] cntr_f_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) lb_re_d <= 1'b0;
        else if (pipe_en) begin lb_re_d <= lb_re; cntr_f_d <= cntr_f; end
        else if (out_ready) lb_re_d <= 1'b0;
    end
    assign out_valid = lb_re_d;
    assign out_data  = max_vec(lb_rdata, h_max_reg[cntr_f_d]);

    function automatic logic [P_CH*A_BIT-1:0] max_vec(input logic [P_CH*A_BIT-1:0] a, input logic [P_CH*A_BIT-1:0] b);
        logic [A_BIT-1:0] a_ch, b_ch;
        for (int unsigned i = 0; i < P_CH; i++) begin
            a_ch                    = a[i*A_BIT+:A_BIT];
            b_ch                    = b[i*A_BIT+:A_BIT];
            max_vec[i*A_BIT+:A_BIT] = (a_ch > b_ch) ? a_ch : b_ch;
        end
    endfunction
endmodule
