// case clr=1,dat_vld=0 acc_r=0

module mac_body #(
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,                        
    input  logic        [A_BIT-1:0] x,
    input  logic signed [W_BIT-1:0] w,
    input  logic signed [B_BIT-1:0] cas_in,
    output logic signed [B_BIT-1:0] cas_out
);
    logic signed [A_BIT  :0] x_ext;
    logic signed [B_BIT-1:0] prod;
    assign x_ext   = {1'b0, x}; // Extend x to signed
    assign prod    = x_ext * w; 

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cas_out <= '0;
        end else if (en) begin
            cas_out <= prod + cas_in;
        end
    end

endmodule

module mac_tail #(
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic                    dat_vld,
    input  logic                    clr,
    input  logic        [A_BIT-1:0] x,
    input  logic signed [W_BIT-1:0] w,
    input  logic signed [B_BIT-1:0] cas_in,
    output logic signed [B_BIT-1:0] acc
);
    logic signed [A_BIT  :0] x_ext;
    logic signed [B_BIT-1:0] prod;
    logic signed [B_BIT-1:0] acc_r;
    assign x_ext = {1'b0, x};
    assign prod = x_ext * w;

     always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_r <= '0;
        end else if (en) begin
            case ({clr, dat_vld})
                2'b00: acc_r <= acc_r;
                2'b01: acc_r <= acc_r + prod + cas_in;
                2'b10: acc_r <= '0; //if dat_vld=0 prod+cas_in=0; if clr=1 acc_r=0; acc_r=0
                2'b11: acc_r <= prod + cas_in;
            endcase
        end
    end

    assign acc = acc_r;
endmodule

module conv_mac_array #(
    parameter int unsigned P_ICH = 4,
    parameter int unsigned A_BIT = 8,
    parameter int unsigned W_BIT = 8,
    parameter int unsigned B_BIT = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic                    dat_vld,
    input  logic                    clr,
    input  logic        [A_BIT-1:0] x_vec [P_ICH],
    input  logic signed [W_BIT-1:0] w_vec [P_ICH],
    output logic signed [B_BIT-1:0] acc
);

    logic signed [B_BIT-1:0] mac_cascade[P_ICH];

    assign mac_cascade[0] = '0;

    logic               [A_BIT-1:0] x_vec_dly [P_ICH];
    logic        signed [W_BIT-1:0] w_vec_dly [P_ICH];
    logic               [P_ICH-1:1]dat_vld_dly;
    logic               [P_ICH-1:1]clr_dly;                
    generate
        for(genvar i = 0; i < P_ICH; i++) begin : gen_x_vec_dly
            delayline # (
                .WIDTH(A_BIT),
                .DEPTH(i)
            ) u_x_vec_dly (
                .clk(clk),
                .rst_n(rst_n),
                .en(en),
                .data_in(x_vec[i]),
                .data_out(x_vec_dly[i])
            );
        end
    endgenerate

    generate
        for(genvar i = 0; i < P_ICH; i++) begin : gen_w_vec_dly
            delayline # (
                .WIDTH(W_BIT),
                .DEPTH(i)
            ) u_w_vec_dly (
                .clk(clk),
                .rst_n(rst_n),
                .en(en),
                .data_in(w_vec[i]),
                .data_out(w_vec_dly[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dat_vld_dly <= '0;
            clr_dly     <= '0;
        end if (en) begin
            dat_vld_dly[1] <= dat_vld;//0-1
            clr_dly[1] <= clr;
            for (int i = 1; i < P_ICH-1; i++) begin
                dat_vld_dly[i+1] <= dat_vld_dly[i];//1-2 2;2-3 3
                clr_dly[i+1] <= clr_dly[i];
            end
        end
    end

    generate
        for (genvar i = 0; i < P_ICH - 1; i++) begin : gen_mac_body
            mac_body #(
                .A_BIT(A_BIT),
                .W_BIT(W_BIT),
                .B_BIT(B_BIT)
            ) u_mac_body (
                .clk    (clk),
                .rst_n  (rst_n),
                .en     (en),
                .x      (x_vec_dly[i]),//0,1,2
                .w      (w_vec_dly[i]),//0,1,2
                .cas_in (mac_cascade[i]),
                .cas_out(mac_cascade[i+1])
            );
        end
    endgenerate

    mac_tail #(
        .A_BIT(A_BIT),
        .W_BIT(W_BIT),
        .B_BIT(B_BIT)
    ) u_mac_tail (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (en),
        .dat_vld(dat_vld_dly[P_ICH-1]),//3
        .clr    (clr_dly[P_ICH-1]),//3
        .x      (x_vec_dly[P_ICH-1]),//3
        .w      (w_vec_dly[P_ICH-1]),//3
        .cas_in (mac_cascade[P_ICH-1]),
        .acc    (acc)
    );

endmodule
