import numpy as np

def ceil_div(x, y):
    return (x + y - 1) // y

def test_matmul():
    M = np.random.randint(3, 100) # generate from 3 to 100
    K = np.random.randint(3, 100)
    N = np.random.randint(3, 100)
    M0 = 3
    N0 = 4
    K0 = 5    
    M1 = ceil_div(M, M0)
    N1 = ceil_div(N, N0)
    K1 = ceil_div(K, K0)
    # 定义矩阵乘法参数
    layer = {
        'M': M,
        'N': N,
        'K': K,
        'M1': M1,
        'N1': N1,
        'K1': K1
    }
    print(layer)

    left_mk = np.random.randint(-128, 127, size=(M, K))
    right_nk = np.random.randint(-128, 127, size=(N, K))
    result_m1n1m0n0 = np.zeros((M1, N1, M0, N0)) # result initial

    left_m1m0k1k0 = np.zeros((M1*M0, K1*K0)) # left m1m0k1k0 initial
    left_m1m0k1k0[:M, :K] = left_mk # left m1m0k1k0 fill
    left_m1k1m0k0 = left_m1m0k1k0.reshape(M1, M0, K1, K0).transpose(0, 2, 1, 3) # move axis

    right_n1n0k1k0 = np.zeros((N1*N0, K1*K0))
    right_n1n0k1k0[:N, :K] = right_nk
    right_n1k1n0k0 = right_n1n0k1k0.reshape(N1, N0, K1, K0).transpose(0, 2, 1, 3)

    for m1 in range(M1): # cycle M1
        for n1 in range(N1): # cycle N1
            temp = np.zeros((M0, N0)) # temp initial
            for k1 in range(K1): # cycle K1
                temp += left_m1k1m0k0[m1][k1] @ right_n1k1n0k0[n1][k1].transpose() # m0k0 @ n0k0^T
            result_m1n1m0n0[m1][n1] = temp # C1(m1n1:unit is 3*4 matrix)=A1B1+A2B2+...(tiled matmul)

    result_mn = result_m1n1m0n0.transpose(0, 2, 1, 3).reshape(M1*M0, N1*N0)[:M, :N] # move axis and reshape
    result_golden = np.matmul(left_mk, right_nk.transpose()) # L_matrix*R_matrix^T
    diff = np.abs(result_golden - result_mn)
    if(diff.sum() == 0):
        print('Pass')

for i in range(10):
    test_matmul()