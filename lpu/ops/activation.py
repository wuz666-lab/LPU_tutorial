from isa import *
from utils import *
from semantic import *

def op_softmax(matrix_mn):
    M_L2 = matrix_mn.shape[0]
    N_L2 = matrix_mn.shape[1]

    N1_L2 = ceil_div(N_L2, N0)

    M_L0 = np.random.randint(3, M_L2)

    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)
    result_mn = torch.zeros((M_L2, N_L2), dtype=matrix_mn.dtype)

    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
        # 因为做softmax必须在N方向做reduce，所以在N方向不切分
        # softmax的输入存放在PSB中，但是没有GM到PSB的数据通路，所以暂时借用gdma_mov2lmb把n1mn0转换成m1n1m0n0
        # 这里只是做指令集验证，后面softmax会放到矩阵乘法后面
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(matrix_n1mn0_l2, M_L2, l0_m_start_in_l2, N1_L2, 0, m_size_l0, N1_L2)
        # max = torch.max(scale_)
        # reduce -> max
        max_m1n1m0n0_arb, = isa.aru(slice_m=m_size_l0, slice_n=N_L2,                         
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en = False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False, 
                      reduce_m_en=False, reduce_n_en=True, reduce_mode=0, # max reduce, 
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        # broadcast max -> sub -> exp | ub -> sum | arb 
        exp_m1n1m0n0_ub, sum_arb, = isa.aru(m_size_l0, N_L2,                   
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=max_m1n1m0n0_arb, arb_en=True, br_m=False, br_n=True, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=True, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=True, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=True, reduce_mode=2, # sum reduce
                      ub_wr_en=True, ub_layout=0, gm_wr_en=False, arb_wr_en=True# 因为exp还要用，所以要输出到ub
                      )
        # output = exp / sum
        result_n1mn0, = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=None, psb_rd_en=False, ub_m1n1m0n0=exp_m1n1m0n0_ub, ub_rd_en=True,
                      arb_in=sum_arb, arb_en=True, br_m=False, br_n=True, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=True, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=-1, # sum reduce
                      ub_wr_en=True, ub_layout=1, gm_wr_en=False, arb_wr_en=False
                      )
        result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, :] = k1mk0_to_mk(result_n1mn0, N_L2)
    return result_mn    

def op_layernorm(matrix_mn):
    M_L2 = matrix_mn.shape[0]
    N_L2 = matrix_mn.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    M_L0 = np.random.randint(3, M_L2)
    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)
    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        l0_m_end_in_l2 = min(l0_m_start_in_l2 + M_L0, M_L2)
        m_size_l0 = l0_m_end_in_l2 - l0_m_start_in_l2
        # layernorm跟softmax类似
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(matrix_n1mn0_l2, m_size_l0, N1_L2, 0, M_L2, N1_L2)
        # mean = torch.mean(scale_)
        mean_m1n1m0n0_arb = isa.aru(slice_m=m_size_l0, slice_n=N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=None, arb_en=False, br_m=False, br_n=None, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=True, reduce_mode=0, # max reduce,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        """
        sub = x - mean
        pow = torch.pow(sub, 2)
        var = torch.mean(pow)
        """
        var_m1m0_arb = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=mean_m1n1m0n0_arb, arb_en=True, br_m=False, br_n=True, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=True, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=True, recp_en=False,
                      reduce_m_en=False, reduce_n_en=True, reduce_mode=3, # mean reduce
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True # 这里的pow不会再用了，所以不用写到UB
                      )
        # sqrt = torch.sqrt(var), 注意，虽然对var算sqrt并不需要broadcast，但是sqrt之后有一个双目操作，需要两路数据，所以这一步要做broadcast
        sqrt_m1m0_arb = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=None, psb_rd_en=False, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=var_m1m0_arb, arb_en=True, br_m=False, br_n=False, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=True,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True # 这里虽然没做reduce, 但是还是把数据写到arb里，方便下一步做broadcast
                      )
        # norm = sub / sqrt
        norm_n1mn0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=sqrt_m1m0_arb, arb_en=True, br_m=False, br_n=True, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=True, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=True, ub_layout=1, gm_wr_en=False
                      )
        result_mn = k1mk0_to_mk(norm_n1mn0_ub, N_L2)
        return result_mn

def op_rmsnorm(matrix_mn):
    M_L2 = matrix_mn.shape[0]
    N_L2 = matrix_mn.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    M_L0 = np.random.randint(3, M_L2)
    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)
    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        l0_m_end_in_l2 = min(l0_m_start_in_l2 + M_L0, M_L2)
        m_size_l0 = l0_m_end_in_l2 - l0_m_start_in_l2
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(matrix_n1mn0_l2, m_size_l0, N1_L2, 0, M_L2, N1_L2)
        # pow = torch.pow(x, 2)
        # mean = torch.mean(pow)
        mean_m1n1m0n0_arb = isa.aru(slice_m=m_size_l0, slice_n=N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=None, arb_en=False, br_m=False, br_n=None, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=True, recp_en=False,
                      reduce_m_en=False, reduce_n_en=True, reduce_mode=3, # mean reduce,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        # sqrt = torch.sqrt(mean)
        rms_m1m0_arb = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=None, psb_rd_en=False, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=mean_m1n1m0n0_arb, arb_en=True, br_m=False, br_n=False, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=True,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0, # sum reduce
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        # norm = sub / sqrt
        norm_n1mn0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=rms_m1m0_arb, arb_en=True, br_m=False, br_n=True, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=True, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0, # sum reduce
                      ub_wr_en=True, ub_layout=1, gm_wr_en=False
                      )
        result_mn = k1mk0_to_mk(norm_n1mn0_ub, N_L2)
        return result_mn

def op_sigmoid(matrix_mn):
    M_L2 = matrix_mn.shape[0]
    N_L2 = matrix_mn.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    M_L0 = np.random.randint(3, M_L2)
    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)
    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        l0_m_end_in_l2 = min(l0_m_start_in_l2 + M_L0, M_L2)
        m_size_l0 = l0_m_end_in_l2 - l0_m_start_in_l2
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(matrix_n1mn0_l2, m_size_l0, N1_L2, 0, M_L2, N1_L2)
        # exp = torch.exp(-x)
        exp_m1n1m0n0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=False, ub_m1n1m0n0=None, ub_rd_en=False,
                      arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en=False, scalar=None, 
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=-500., clamp_max=500., exp_en=True, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=True, ub_layout=0, gm_wr_en=False, arb_wr_en=True# 因为exp还要用，所以要输出到ub
                      )
        # sigmoid = 1 / (1 + exp) 这里的1怎么给进去有点纠结，在指令里直接给，还是先把数据给到UB，然后把数据从UB搬运到ARB，这么做太麻烦了。
        # 感觉可以在指令域段里给一个scalar域段，binary操作可以选择从scalar获取一路数据，这样比较符合直觉，缺点是每个binary都要增加一个域段，标识数据从哪里进
        sigmoid_m1n1m0n0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=None, psb_rd_en=False, ub_m1n1m0n0=sigmoid_m1n1m0n0_ub, ub_rd_en=True,
                      arb_in=exp_m1n1m0n0_ub, arb_en=False, br_m=False, br_n=False, scalar_en=True, scalar=1., 
                      add_en=True, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, 
                      neg_en=False, clamp_en=False, clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        result_mn = k1mk0_to_mk(sigmoid_m1n1m0n0_ub, N_L2)
        return result_mn

def op_silu(matrix_mn):
    M_L2 = matrix_mn.shape[0]
    N_L2 = matrix_mn.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    M_L0 = np.random.randint(3, M_L2)
    isa = ISA()
    matrix_n1mn0_l2 = mk_to_k1mk0(matrix_mn)
    for l0_m_start_in_l2 in range(0, M_L2, M_L0):
        l0_m_end_in_l2 = min(l0_m_start_in_l2 + M_L0, M_L2)
        m_size_l0 = l0_m_end_in_l2 - l0_m_start_in_l2
        matrix_m1n1m0n0_l0 = isa.gdma_mov2lmb(matrix_n1mn0_l2, m_size_l0, N1_L2, 0, M_L2, N1_L2)
        # exp = torch.exp(-x)
        exp_m1n1m0n0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_m1n1m0n0_l0, psb_rd_en=False, ub_m1n1m0n0=None, ub_rd_en=False, scalar_en=False, scalar=None, 
                      arb_in=None, arb_en=False, br_m=False, br_n=False,
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=-500., clamp_max=500., exp_en=True, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=True, ub_layout=0, gm_wr_en=False, arb_wr_en=True# 因为exp还要用，所以要输出到ub
                      )
        # sigmoid = 1 / (1 + exp) 这里的1怎么给进去有点纠结，在指令里直接给，还是先把数据给到UB，然后把数据从UB搬运到ARB，这么做太麻烦了。
        # 感觉可以在指令域段里给一个scalar域段，binary操作可以选择从scalar获取一路数据，这样比较符合直觉，缺点是每个binary都要增加一个域段，标识数据从哪里进
        sigmoid_m1n1m0n0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=None, psb_rd_en=False, ub_m1n1m0n0=exp_m1n1m0n0_ub, ub_rd_en=True, scalar_en=True, scalar=1., 
                      arb_in=None, arb_en=False, br_m=False, br_n=False,
                      add_en=True, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        silu_n1mn0_ub = isa.aru(m_size_l0, N_L2,
                      psb_m1n1m0n0=matrix_n1mn0_l2, psb_rd_en=True, ub_m1n1m0n0=sigmoid_m1n1m0n0_ub, ub_rd_en=True, scalar_en=False, scalar=None, 
                      arb_in=None, arb_en=False, br_m=False, br_n=False,
                      add_en=False, sub_en=False, max_en=False, min_en=False,
                      mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                      clamp_min=None, clamp_max=None, exp_en=False, sqrt_en=False,
                      pow_en=False, recp_en=False,
                      reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                      ub_wr_en=False, ub_layout=0, gm_wr_en=False, arb_wr_en=True
                      )
        result_mn = k1mk0_to_mk(silu_n1mn0_ub, N_L2)
        return result_mn