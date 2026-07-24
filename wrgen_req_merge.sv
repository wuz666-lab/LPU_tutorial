module wrgen_req_merge #(
    parameter int MERGE_NUM  = 2,
    parameter int DATA_WIDTH = 512,
    parameter int MASK_WIDTH = 64,
    parameter int ADDR_WIDTH = 34,
    parameter int ADDR_STEP  = 64
) (
    input  logic                                    clk,
    input  logic                                    rst_n,

    // Input Interface
    input  logic                                    in_vld,
    input  logic [ADDR_WIDTH-1:0]                   in_addr,
    input  logic [DATA_WIDTH-1:0]                   in_dat,
    input  logic [MASK_WIDTH-1:0]                   in_msk,
    input  logic                                    in_lst,
    output logic                                    in_rdy,

    // Output Interface
    output logic                                    out_vld,
    output logic [ADDR_WIDTH-1:0]                   out_addr,
    output logic [MERGE_NUM*DATA_WIDTH-1:0]         out_dat,
    output logic [MERGE_NUM*MASK_WIDTH-1:0]         out_msk,
    output logic                                    out_lst,
    input  logic                                    out_rdy
);

    localparam int BUFFER_LENGTH = MERGE_NUM - 1;
    localparam int CNT_WIDTH = (BUFFER_LENGTH > 1) ? $clog2(BUFFER_LENGTH + 1) : 1;
    /*---------------------------submodule----------------------*/
    logic [BUFFER_LENGTH-1:0][DATA_WIDTH-1:0] data_buffer;          // 缓存输入数据
    logic [BUFFER_LENGTH-1:0][MASK_WIDTH-1:0] mask_buffer;          // 缓存输入掩码
    logic [ADDR_WIDTH-1:0] last_addr_reg;                           // 记录上一个输入地址

    logic [ADDR_WIDTH-1:0] batch_addr_reg;                          // 记录当前批次的起始地址
    logic [CNT_WIDTH-1:0] cnt;

    logic out_vld_buffer;                                           // 输出有效信号缓存
    logic out_lst_buffer;                                           // 输出最后一个信号缓存
    logic [MERGE_NUM*DATA_WIDTH-1:0] out_dat_buffer;
    logic [MERGE_NUM*MASK_WIDTH-1:0] out_msk_buffer;
    logic [ADDR_WIDTH-1:0]           out_addr_buffer;
    /*---------------------------end----------------------------*/
    /*---------------------------trigger signal-----------------*/
    logic pipe_en_in;
    logic pipe_en_out;
    logic is_lst_dat;
    logic is_not_continuous;
    logic is_buffer_full;
    
    logic is_continuous_and_lst;                                     // case1: 连续且最后一个  
    logic is_not_continuous_and_not_lst;                             // case2: 不连续且不是最后一个
    logic is_not_continuous_and_lst;                                 // case3: 不连续且是最后一个
    logic is_buffer_full_and_normal;                                 // case4: 缓存满、不是最后一个且连续

    logic concat_and_send;                                           // case1 and case4
    logic send_and_store;                                            // case2
    logic send_twice;                                                // case3
    logic flush_pending;                                             // case3: flush pending for next cycle
    /*-------------------------end-----------------------------*/
    assign pipe_en_in  = in_vld && in_rdy;
    assign pipe_en_out = !out_vld || out_rdy;

    assign is_lst_dat = in_lst;
    assign is_not_continuous = (cnt != 0) && (in_addr != last_addr_reg + ADDR_STEP); // 存在已缓存数据且不连续
    assign is_buffer_full = (cnt == BUFFER_LENGTH);                                 
 
    assign is_continuous_and_lst = !is_not_continuous && is_lst_dat;
    assign is_not_continuous_and_not_lst = is_not_continuous && !is_lst_dat;
    assign is_not_continuous_and_lst = is_not_continuous && is_lst_dat;
    assign is_buffer_full_and_normal = is_buffer_full && !is_not_continuous && !is_lst_dat;

    assign concat_and_send = is_continuous_and_lst || is_buffer_full_and_normal;
    assign send_and_store = is_not_continuous_and_not_lst;
    assign send_twice = is_not_continuous_and_lst;

    assign in_rdy = pipe_en_out;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_addr_reg <= '0;
        end else if (pipe_en_in) begin
            last_addr_reg <= in_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch_addr_reg <= '0;
        end else if (pipe_en_in &&
                     ((cnt == 0) || send_and_store || send_twice || flush_pending)) begin
            batch_addr_reg <= in_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= '0;
        end else if (pipe_en_out && flush_pending) begin
            cnt <= pipe_en_in ? 1'b1 : '0;
        end else if (pipe_en_in) begin
            if (concat_and_send || flush_pending) begin
                cnt <= '0;
            end else if (send_and_store || send_twice) begin
                cnt <= 1'b1;
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            flush_pending <= 1'b0;
        end else if (pipe_en_out && flush_pending) begin
            flush_pending <= pipe_en_in && is_lst_dat;
        end else if (pipe_en_in && send_twice) begin
            flush_pending <= 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_buffer <= '0;
            mask_buffer <= '0;
        end else if (pipe_en_in) begin
            if (send_and_store || send_twice || flush_pending) begin
                data_buffer[0] <= in_dat;
                mask_buffer[0] <= in_msk;
            end else begin
                data_buffer[cnt] <= in_dat;
                mask_buffer[cnt] <= in_msk;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_vld_buffer <= 1'b0;
            out_lst_buffer <= 1'b0;
        end else if (pipe_en_out) begin
            out_vld_buffer <= 1'b0;
            if (flush_pending) begin
                out_vld_buffer <= 1'b1;
                out_lst_buffer <= 1'b1;
            end else if (pipe_en_in) begin
                if (concat_and_send) begin
                    out_vld_buffer <= 1'b1;
                    out_lst_buffer <= is_lst_dat;
                end else if (send_and_store || send_twice) begin
                    out_vld_buffer <= 1'b1;
                    out_lst_buffer <= 1'b0;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_addr_buffer <= '0;
        end else if (pipe_en_out) begin
            if (flush_pending) begin
                out_addr_buffer <= batch_addr_reg;
            end else if (pipe_en_in) begin
                if (concat_and_send) begin
                    out_addr_buffer <= (cnt == 0) ? in_addr : batch_addr_reg;
                end else if (send_and_store || send_twice) begin
                    out_addr_buffer <= batch_addr_reg;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_dat_buffer <= '0;
            out_msk_buffer <= '0;
        end else if (pipe_en_out) begin
            out_dat_buffer <= '0;
            out_msk_buffer <= '0;
            if (flush_pending) begin
                for (int entry_index = 0; entry_index < BUFFER_LENGTH;
                     entry_index = entry_index + 1) begin
                    if (entry_index < cnt) begin
                        out_dat_buffer[entry_index*DATA_WIDTH+:DATA_WIDTH] <=
                            data_buffer[entry_index];
                        out_msk_buffer[entry_index*MASK_WIDTH+:MASK_WIDTH] <=
                            mask_buffer[entry_index];
                    end
                end
            end else if (pipe_en_in) begin
                if (concat_and_send) begin
                    for (int entry_index = 0; entry_index < BUFFER_LENGTH;
                         entry_index = entry_index + 1) begin
                        if (entry_index < cnt) begin
                            out_dat_buffer[entry_index*DATA_WIDTH+:DATA_WIDTH] <=
                                data_buffer[entry_index];
                            out_msk_buffer[entry_index*MASK_WIDTH+:MASK_WIDTH] <=
                                mask_buffer[entry_index];
                        end
                    end
                    out_dat_buffer[cnt*DATA_WIDTH+:DATA_WIDTH] <= in_dat;
                    out_msk_buffer[cnt*MASK_WIDTH+:MASK_WIDTH] <= in_msk;
                end else if (send_and_store || send_twice) begin
                    for (int entry_index = 0; entry_index < BUFFER_LENGTH;
                         entry_index = entry_index + 1) begin
                        if (entry_index < cnt) begin
                            out_dat_buffer[entry_index*DATA_WIDTH+:DATA_WIDTH] <=
                                data_buffer[entry_index];
                            out_msk_buffer[entry_index*MASK_WIDTH+:MASK_WIDTH] <=
                                mask_buffer[entry_index];
                        end
                    end
                end
            end
        end
    end

    assign out_vld = out_vld_buffer;
    assign out_lst = out_lst_buffer;
    assign out_dat = out_dat_buffer;
    assign out_msk = out_msk_buffer;
    assign out_addr = out_addr_buffer;

endmodule
