module add #(
    parameter P_CH  = 4,
    parameter A_BIT = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in1_valid,
    output logic                  in1_ready,
    input  logic [P_CH*A_BIT-1:0] in1_data,
    input  logic                  in2_valid,
    output logic                  in2_ready,
    input  logic [P_CH*A_BIT-1:0] in2_data,
    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [P_CH*A_BIT-1:0] out_data
);

    logic                  pipe_valid;
    logic [P_CH*A_BIT-1:0] pipe_data;
    logic                  handshake_in;
    logic                  handshake_out;
    logic [P_CH*A_BIT-1:0] calc_result;

    assign handshake_in  = in1_valid && in1_ready && in2_valid && in2_ready;
    assign handshake_out = out_valid && out_ready;
    

    assign in1_ready     = in2_valid&&(!pipe_valid || out_ready);
    assign in2_ready     = in1_valid&&(!pipe_valid || out_ready);

    always_comb begin
        for (int i = 0; i < P_CH; i++) begin
            logic unsigned [A_BIT-1:0] x1;
            logic unsigned [A_BIT-1:0] x2;
            logic unsigned [A_BIT-1:0] sum;

            x1                          = in1_data[i*A_BIT+:A_BIT];
            x2                          = in2_data[i*A_BIT+:A_BIT];
            sum                         = x1 + x2;

            calc_result[i*A_BIT+:A_BIT] = sum;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_valid <= 1'b0;
            pipe_data  <= '0;
        end else begin
            if(pipe_valid == 1'b1)begin
                if (out_ready) begin
                    if(in1_valid && in2_valid) begin
                        pipe_valid <= 1;
                        pipe_data  <= calc_result;
                    end else begin
                    pipe_valid <= 0;
                    end
                end 
            end else begin
                if (in1_valid && in2_valid) begin
                    pipe_valid <= 1;
                    pipe_data <= calc_result;
                end
            end
        end
    end
    assign out_valid = pipe_valid;
    assign out_data  = pipe_data;

endmodule
