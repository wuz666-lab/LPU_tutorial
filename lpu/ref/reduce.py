import numpy as np
from utils import *
from common import *

M = 11
N = 9
M1 = ceil_div(M, M0)
N1 = ceil_div(N, N0)
P_ARU = 2
mn = np.arange(M * N).reshape((M, N)).astype(np.float32)
m1m0n1n0 = np.zeros((M1*M0, N1*N0), dtype=mn.dtype)
m1m0n1n0[:M, :N] = mn
mn_m1n1m0n0 = m1m0n1n0.reshape(M1, M0, N1, N0).transpose(0, 2, 1, 3)
mn_reduce_n = np.max(mn, axis=1)
mn_reduce_m = np.max(mn, axis=0)

def reduce_m0n0(mn_m1n1m0n0, reduce_m_en, reduce_n_en):
    reduce_n_result = np.zeros((M1, M0), dtype=mn_m1n1m0n0.dtype)
    reduce_m_result = np.zeros((N1, N0), dtype=mn_m1n1m0n0.dtype)
    reduce_n_temp = np.zeros((P_ARU), dtype=mn_m1n1m0n0.dtype)
    reduce_m_temp = np.zeros((N0), dtype=mn_m1n1m0n0.dtype)
    reduce_mn_result = -np.inf
    if reduce_n_en and reduce_m_en:
        for m1 in range(M1):
            for m0 in range(M0 // P_ARU):
                reduce_n_temp.fill(-np.inf)  # Initialize to negative infinity
                for n1 in range(N1):
                    input = mn_m1n1m0n0[m1, n1, m0*P_ARU:(m0+1)*P_ARU]
                    for p in range(P_ARU):
                        for n0 in range(N0):
                            reduce_mn_result = max(input[p, n0], reduce_mn_result)
        return reduce_mn_result
    elif reduce_n_en:
        for m1 in range(M1):
            for m0 in range(M0 // P_ARU):
                reduce_n_temp.fill(-np.inf)  # Initialize to negative infinity
                for n1 in range(N1):
                    input = mn_m1n1m0n0[m1, n1, m0*P_ARU:(m0+1)*P_ARU]
                    for p in range(P_ARU):
                        for n0 in range(N0):
                            reduce_n_temp[p] = max(input[p, n0], reduce_n_temp[p])
                reduce_n_result[m1, m0*P_ARU:(m0+1)*P_ARU] = reduce_n_temp
        return reduce_n_result
    elif reduce_m_en:
        for n1 in range(N1):
            for m1 in range(M1):
                reduce_m_temp.fill(-np.inf)  # Initialize to negative infinity
                for m0 in range(M0 // P_ARU):
                    input = mn_m1n1m0n0[m1, n1, m0*P_ARU:(m0+1)*P_ARU]
                    for p in range(P_ARU):
                        for n0 in range(N0):
                            reduce_m_temp[n0] = max(input[p, n0], reduce_m_temp[n0])
            reduce_m_result[n1] = reduce_m_temp
        return reduce_m_result

def reduce_n1n0(mn_m1n1m0n0, reduce_m_en, reduce_n_en):
    reduce_n_result = np.zeros((M1, M0), dtype=mn_m1n1m0n0.dtype)
    reduce_m_result = np.zeros((N1, N0), dtype=mn_m1n1m0n0.dtype)

    reduce_m_temp = np.zeros((N0), dtype=mn_m1n1m0n0.dtype)

    if reduce_n_en and reduce_m_en:
        reduce_mn_result = -np.inf
        for m1 in range(M1):
            for m0 in range(M0):
                for n1 in range(ceil_div(N1, P_ARU)):
                    input = mn_m1n1m0n0[m1, n1*P_ARU:(n1+1)*P_ARU, m0]
                    for p in range(P_ARU):
                        for n0 in range(N0):
                            reduce_mn_result = max(input[p, n0], reduce_mn_result)
        return reduce_mn_result
    elif reduce_n_en:
        for m1 in range(M1):
            for m0 in range(M0):
                reduce_n_temp = -np.inf
                for n1 in range(ceil_div(N1, P_ARU)):
                    input = mn_m1n1m0n0[m1, n1*P_ARU:(n1+1)*P_ARU, m0]
                    for p in range(P_ARU):
                        for n0 in range(N0):
                            reduce_n_temp = max(input[p, n0], reduce_n_temp)
                reduce_n_result[m1, m0] = reduce_n_temp
    return reduce_n_result


print("mn:", mn)
print("mn_m1n1m0n0:", mn_m1n1m0n0)
print("mn_reduce_n:", mn_reduce_n)

reduce_n_m1n1m0n0 = reduce_m0n0(mn_m1n1m0n0, False, True)
print("reduce_n_m1n1m0n0:", reduce_n_m1n1m0n0)

print("mn_reduce_m:", mn_reduce_m)
reduce_m_m1n1m0n0 = reduce_m0n0(mn_m1n1m0n0, True, False)
print("reduce_m_m1n1m0n0:", reduce_m_m1n1m0n0)

reduce_mn_m1n1m0n0 = reduce_m0n0(mn_m1n1m0n0, True, True)
print("reduce_mn_m1n1m0n0:", reduce_mn_m1n1m0n0)

reduce_n1n0_m1n1m0n0 = reduce_n1n0(mn_m1n1m0n0, False, True)
print("reduce_n_n1n0_m1n1m0n0:", reduce_n1n0_m1n1m0n0)