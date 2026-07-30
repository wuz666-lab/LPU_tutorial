from isa import *
from utils import *
from semantic import *

def find_optimal_l1_tiling(M_L2, N_L2, K_L2):
    M_L1 = np.random.randint(2, M_L2)
    N_L1 = np.random.randint(2, N_L2)
    K_L1 = np.random.randint(2, K_L2)
    return M_L1, N_L1, K_L1

def find_optimal_l0_tiling(M_L1, N_L1, K_L1):
    # randint 的上界不包含；允许 L0 与 L1 等大以覆盖 L1 恰好为 2 的情况。
    M_L0 = np.random.randint(2, M_L1 + 1)
    N_L0 = np.random.randint(2, N_L1 + 1)
    K_L0 = np.random.randint(2, K_L1 + 1)
    return M_L0, N_L0, K_L0

def op_matmul_tile_twice(left, right, bias):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert K_L2 % K0 == 0
    assert K_L2 == right.shape[1]
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    K1_L1 = ceil_div(K_L1, K0)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    K1_L0 = ceil_div(K_L0, K0)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    isa = ISA()
    left_mk1k0_l2 = torch.zeros((M_L2, K1_L2 * K0), dtype=left.dtype)
    left_mk1k0_l2[:, :] = left
    left_k1mk0_l2 = left_mk1k0_l2.reshape(M_L2, K1_L2, K0).permute(1, 0, 2)
    right_nk1k0_l2 = torch.zeros((N_L2, K1_L2 * K0), dtype=right.dtype)
    right_nk1k0_l2[:, :] = right
    right_k1nk0_l2 = right_nk1k0_l2.reshape(N_L2, K1_L2, K0).permute(1, 0, 2)
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)
    for l1_n_start_in_l2 in range(0, N_L2, N_L1):
        n_size_l1 = min(N_L1, N_L2 - l1_n_start_in_l2)
        bias_l1 = bias[l1_n_start_in_l2:l1_n_start_in_l2 + n_size_l1]
        for l1_m_start_in_l2 in range(0, M_L2, M_L1):
            m_size_l1 = min(M_L1, M_L2 - l1_m_start_in_l2)
            #psb: tile unit
            result_m2n2m1n1m0n0_psb = torch.zeros((ceil_div(m_size_l1, M_L0)*ceil_div(n_size_l1, N_L0), M1_L0, N1_L0, M0, N0))
            for l1_k1_start_in_l2 in range(0, K1_L2, K1_L1):
                k1_size_l1 = min(K1_L1, K1_L2 - l1_k1_start_in_l2)
                left_k1mk0_l1 = isa.gdma_mov2ub(left_k1mk0_l2, M_L2, l1_m_start_in_l2, K1_L2, l1_k1_start_in_l2, m_size_l1, k1_size_l1)
                right_k1nk0_l1 = isa.gdma_mov2ub(right_k1nk0_l2, N_L2, l1_n_start_in_l2, K1_L2, l1_k1_start_in_l2, n_size_l1, k1_size_l1)
                for l0_n_start_in_l1 in range(0, n_size_l1, N_L0):
                    n_size_l0 = min(N_L0, n_size_l1 - l0_n_start_in_l1)
                    bias_n1n0_pmb = isa.gdma_mov2pmb(bias_l1, n_size_l1, l0_n_start_in_l1, n_size_l0)
                    for l0_m_start_in_l1 in range(0, m_size_l1, M_L0):
                        m_size_l0 = min(M_L0, m_size_l1 - l0_m_start_in_l1)
                        # slice unit
                        result_n1mn0 = torch.zeros((N1_L0, m_size_l0, N0), dtype=left.dtype)

                        for l0_k1_start_in_l1 in range(0, k1_size_l1, K1_L0):
                            k1_size_l0 = min(K1_L0, k1_size_l1 - l0_k1_start_in_l1)        
                            left_m1k1m0k0_l0 = isa.ldma_mov2lmb(left_k1mk0_l1, m_size_l1, l0_m_start_in_l1, k1_size_l1, l0_k1_start_in_l1, m_size_l0, k1_size_l0)
                            right_n1k1n0k0_l0 = isa.ldma_mov2rmb(right_k1nk0_l1, n_size_l1, l0_n_start_in_l1, k1_size_l1, l0_k1_start_in_l1, n_size_l0, k1_size_l0)
                            bias_en = (l0_k1_start_in_l1 == 0 and l1_k1_start_in_l2 == 0)
                            psum_en = not bias_en
                            output_en = (l0_k1_start_in_l1 + k1_size_l0 == k1_size_l1) and (l1_k1_start_in_l2 + k1_size_l1 == K1_L2)
                            # slice idx
                            psb_idx = (l0_m_start_in_l1//M_L0 * ceil_div(n_size_l1, N_L0) + l0_n_start_in_l1//N_L0)
                            # L0 -> mxu (slice)
                            result_m2n2m1n1m0n0_psb[psb_idx] = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m2n2m1n1m0n0_psb[psb_idx], m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype='fp16')

                            if output_en:
                                result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                                    result_m2n2m1n1m0n0_psb[psb_idx], psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en=False, scalar=None, \
                                                    add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                                    neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                                    reduce_m_en=False, reduce_n_en=False, reduce_mode=None, ub_wr_en=True, ub_layout=1, gm_wr_en=False, arb_wr_en=False)
                                result_mn[
                                    l1_m_start_in_l2 + l0_m_start_in_l1:l1_m_start_in_l2 + l0_m_start_in_l1 + m_size_l0,
                                    l1_n_start_in_l2 + l0_n_start_in_l1:l1_n_start_in_l2 + l0_n_start_in_l1 + n_size_l0,
                                ] = k1mk0_to_mk(result_n1mn0[0], n_size_l0)
    return result_mn

def op_matmul_tile_once(left, right, bias):
    M_L2 = left.shape[0] # MK
    N_L2 = right.shape[0] # NK
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[1])
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    K1_L0 = ceil_div(K_L0, K0)

    isa = ISA()

    left_mk1k0_l2 = torch.zeros((M_L2, K1_L2*K0))
    left_mk1k0_l2[:, :] = left
    left_k1mk0_l2 = left_mk1k0_l2.reshape(M_L2, K1_L2, K0).permute(1, 0, 2)
    right_nk1k0_l2 = torch.zeros((N_L2, K1_L2*K0))
    right_nk1k0_l2[:, :] = right
    right_k1nk0_l2 = right_nk1k0_l2.reshape(N_L2, K1_L2, K0).permute(1, 0, 2)

    result_n1mn0 = torch.zeros((N1_L0, M_L0, N0), dtype=left.dtype)
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)
    result_m1n1m0n0_l0 = torch.zeros((M1_L0, N1_L0, M0, N0), dtype=left.dtype)

    for l0_n_start_in_l2 in range(0, N_L2, N_L0):
        n_size_l0 = min(N_L0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, N_L2, l0_n_start_in_l2, n_size_l0)
        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)
            for l0_k1_start_in_l2 in range(0, K1_L2, K1_L0):
                k1_size_l0 = min(K1_L0, K1_L2 - l0_k1_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, M_L2, l0_m_start_in_l2, K1_L2, l0_k1_start_in_l2, m_size_l0, k1_size_l0)
                right_n1k1n0k0_l0 = isa.gdma_mov2rmb(right_k1nk0_l2, N_L2, l0_n_start_in_l2, K1_L2, l0_k1_start_in_l2, n_size_l0, k1_size_l0)
                bias_en = l0_k1_start_in_l2 == 0 #fst
                psum_en = l0_k1_start_in_l2 > 0 #middle
                output_en = (l0_k1_start_in_l2 // K1_L0 == ceil_div(K1_L2, K1_L0) - 1) #lst
                
                result_m1n1m0n0_l0 = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype='fp16')
                if output_en:
                    result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                           result_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en=False, scalar=None, \
                                           add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                           neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                           reduce_m_en=False, reduce_n_en=False, reduce_mode=None, ub_wr_en=True, ub_layout=1, gm_wr_en=False, arb_wr_en=False)
                    result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, l0_n_start_in_l2:l0_n_start_in_l2 + n_size_l0] = k1mk0_to_mk(result_n1mn0[0], n_size_l0)
    return result_mn

def op_matmul_transpose(left, right, bias):
    """
    该算子用于实现 QK^T                                             
    """
    M_L2 = left.shape[0] ## MK
    N_L2 = right.shape[1] ## KN
    N1_L2 = ceil_div(N_L2, N0)
    K_L2 = left.shape[1]
    K1_L2 = ceil_div(K_L2, K0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[0]) # KN  # 这里跟op_matmul_tile_twice不同，右矩阵的layout是k1nk0，而不是n1kn0，做的时候要转置
    M_L1, N_L1, K_L1 = find_optimal_l1_tiling(M_L2, N_L2, K_L2)
    M_L0, N_L0, K_L0 = find_optimal_l0_tiling(M_L1, N_L1, K_L1)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    K1_L0 = ceil_div(K_L0, K0)

    isa = ISA()
    left_mk1k0_l2 = torch.zeros((M_L2, K1_L2*K0))
    left_mk1k0_l2[:M_L2, :K_L2] = left
    left_k1mk0_l2 = left_mk1k0_l2.reshape(M_L2, K1_L2, K0).permute(1, 0, 2)
    right_kn1n0_l2 = torch.zeros((K_L2, N1_L2*N0))
    right_kn1n0_l2[:K_L2, :N_L2] = right
    right_n1kn0_l2 = right_kn1n0_l2.reshape(K_L2, N1_L2, N0).permute(1, 0, 2)

    # slice unit
    result_m1n1m0n0_l0 = torch.zeros((M1_L0, N1_L0, M0, N0), dtype=left.dtype)
     # slice unit 
    result_n1mn0 = torch.zeros((N1_L0, M_L0, N0), dtype=left.dtype)
    result_mn = torch.zeros((M_L2, N_L2), dtype=left.dtype)

    for l0_n1_start_in_l2 in range(0, N1_L2, N1_L0):
        n1_size_l0 = min(N1_L0, N1_L2 - l0_n1_start_in_l2)
        l0_n_start_in_l2 = l0_n1_start_in_l2 * N0
        n_size_l0 = min(n1_size_l0 * N0, N_L2 - l0_n_start_in_l2)
        bias_n1n0_pmb = isa.gdma_mov2pmb(bias, N_L2, l0_n_start_in_l2, n_size_l0)

        for l0_m_start_in_l2 in range(0, M_L2, M_L0):
            m_size_l0 = min(M_L0, M_L2 - l0_m_start_in_l2)

            for l0_k1_start_in_l2 in range(0, K1_L2, K1_L0):
                k1_size_l0 = min(K1_L0, K1_L2 - l0_k1_start_in_l2)
                l0_k_start_in_l2 = l0_k1_start_in_l2 * K0
                k_size_l0 = min(k1_size_l0 * K0, K_L2 - l0_k_start_in_l2)
                left_m1k1m0k0_l0 = isa.gdma_mov2lmb(left_k1mk0_l2, M_L2, l0_m_start_in_l2, K1_L2, l0_k1_start_in_l2, m_size_l0, k1_size_l0)
                right_n1k1n0k0_l0 = isa.ldma_mov2rmb_transpose(right_n1kn0_l2, N1_L2, l0_n1_start_in_l2, K_L2, l0_k_start_in_l2, n1_size_l0, k_size_l0)

                bias_en = l0_k1_start_in_l2 == 0
                psum_en = l0_k1_start_in_l2 > 0
                output_en = (l0_k1_start_in_l2 // K1_L0 == ceil_div(K1_L2, K1_L0) - 1)

                result_m1n1m0n0_l0 = isa.mxu_matmul(left_m1k1m0k0_l0, right_n1k1n0k0_l0, bias_n1n0_pmb, result_m1n1m0n0_l0, m_size_l0, n_size_l0, k1_size_l0, bias_en, psum_en, dtype='fp16')

                if output_en:
                    result_n1mn0 = isa.aru(m_size_l0, n_size_l0, \
                                            result_m1n1m0n0_l0, psb_rd_en=True, ub_m1n1m0n0=None, ub_rd_en=False, arb_in=None, arb_en=False, br_m=False, br_n=False, scalar_en=False, scalar=None, \
                                            add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False, \
                                            neg_en=False, clamp_en=False, clamp_min=-np.inf, clamp_max=np.inf, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False, \
                                            reduce_m_en=False, reduce_n_en=False, reduce_mode=None, ub_wr_en=True, ub_layout=1, gm_wr_en=False, arb_wr_en=False)
                    result_mn[l0_m_start_in_l2:l0_m_start_in_l2 + m_size_l0, l0_n_start_in_l2:l0_n_start_in_l2 + n_size_l0] = k1mk0_to_mk(result_n1mn0[0], n_size_l0)
    return result_mn