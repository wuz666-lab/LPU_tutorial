// module mac_array #(
//     parameter int unsigned P_ICH = 4,
//     parameter int unsigned A_BIT = 8,
//     parameter int unsigned W_BIT = 8,
//     parameter int unsigned B_BIT = 32
// ) (
//     input  logic                    clk,
//     input  logic                    rst_n,
//     input  logic                    en,
//     input  logic                    dat_vld,
//     input  logic                    clr,
//     input  logic        [A_BIT-1:0] x_vec  [P_ICH],
//     input  logic signed [W_BIT-1:0] w_vec  [P_ICH],
//     output logic signed [B_BIT-1:0] acc
// );

//     logic signed [B_BIT-1:0] mac_cascade[P_ICH+1];

//     assign mac_cascade[0] = '0;

//     generate
//         for (genvar i = 0; i < P_ICH - 1; i++) begin : gen_mac
//             logic signed [B_BIT-1:0] acc_r;
//             always_ff @(posedge clk or negedge rst_n) begin
//                 if (!rst_n) begin
//                     acc_r <= '0;
//                 end else if (en) begin
//                     if (dat_vld) begin
//                         acc_r <= ($signed(x_vec[i]) * w_vec[i]) - mac_cascade[i];
//                     end else if (clr) begin
//                         acc_r <= '0;
//                     end
//                 end
//             end
//             assign mac_cascade[i+1] = acc_r;
//         end
//     endgenerate

//     logic signed [B_BIT-1:0] tail_acc_r;
//     always_ff @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             tail_acc_r <= '0;
//         end else if (en) begin
//             if (dat_vld) begin
//                 if (clr) begin
//                     tail_acc_r <= ($signed(x_vec[P_ICH-1]) * w_vec[P_ICH-1]) + mac_cascade[P_ICH-1];
//                 end else begin
//                     tail_acc_r <= tail_acc_r + ($signed(x_vec[P_ICH-1]) * w_vec[P_ICH-1]) + mac_cascade[P_ICH-1];
//                 end
//             end else if (clr) begin
//                 tail_acc_r <= tail_acc_r;
//             end
//         end
//     end
//     assign mac_cascade[P_ICH] = tail_acc_r;

//     assign acc = mac_cascade[P_ICH];

// endmodule
