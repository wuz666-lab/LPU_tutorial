from common import *

def mk_to_k1mk0(mk):
    M = mk.shape[0]
    K = mk.shape[1]
    K1 = ceil_div(K, K0)
    mk1k0 = torch.zeros((M, K1*K0), dtype=mk.dtype)
    mk1k0[:, :K] = mk
    k1mk0 = mk1k0.reshape(M, K1, K0).permute(1, 0, 2)
    return k1mk0

def k1mk0_to_m1k1m0k0(k1mk0):
    M = k1mk0.shape[1]
    M1 = ceil_div(M, M0)
    K1 = k1mk0.shape[0]
    K0 = k1mk0.shape[2]
    k1m1m0k0 = torch.zeros((K1, M1*M0, K0), dtype=k1mk0.dtype)
    k1m1m0k0[:,M] = k1mk0
    m1k1m0k0 = k1mk0.reshape(K1, M1, M0, K0).permute(1, 0, 2, 3)
    return m1k1m0k0

def k1mk0_to_mk(k1mk0, k):
    mk = torch.zeros((k1mk0.shape[1], k), dtype=k1mk0.dtype)
    mk1k0 = k1mk0.permute(1, 0, 2).reshape(k1mk0.shape[1], k1mk0.shape[0]*k1mk0.shape[2])
    mk= mk1k0[:, :k]
    return mk

def m1k1m0k0_to_k1mk0(m1k1m0k0, m):
    # 输入布局为 M1-K1-M0-K0；M1 与 K1 的维度顺序不可交换。
    M1 = m1k1m0k0.shape[0]
    K1 = m1k1m0k0.shape[1]
    M0 = m1k1m0k0.shape[2]
    K0 = m1k1m0k0.shape[3]
    k1m1m0k0 = m1k1m0k0.permute(1, 0, 2, 3).reshape(K1, M1*M0, K0)
    k1mk0 = k1m1m0k0[:, :m] 
    return k1mk0

def generate_matmul_params():
    M_L2 = np.random.randint(3, 100)
    N_L2 = np.random.randint(3, 100)
    K_L2 = ceil_align(np.random.randint(3, 100), K0)
    matmul_params = {
        'M': M_L2,
        'N': N_L2,
        'K': K_L2,
    }
    print(matmul_params)
    return matmul_params

def generate_matmul_tensor(matmul_params):
    M = matmul_params['M']
    N = matmul_params['N']
    K = matmul_params['K']
    left = torch.randn((M, K))
    right = torch.randn((N, K))
    bias = torch.randn((N))
    return left, right, bias

def compare(tensor_test, tensor_golden, int = False, golden_threshold = 0.001, error_threshold = 0.1):
    if int:
        diff = torch.abs(tensor_test.flatten() - tensor_golden.flatten())
        error_rate = diff / (torch.abs(tensor_golden.flatten()) + 1)
        print("average error rate: {:.4f}%".format(error_rate.mean()*100))
        if error_rate.mean() < error_threshold:
            print("pass")
            return True
        else:
            print("fail")
            return False
    else:
        # 受限于float32的计算精度，golden的值太小，相对误差可能很大
        golden_mask = torch.abs(tensor_golden) > golden_threshold
        diff = torch.abs(tensor_test[golden_mask].flatten() - tensor_golden[golden_mask].flatten())
        error_rate = diff / (torch.abs(tensor_golden[golden_mask].flatten()) + 1e-10)
        print("average error rate: {:.4f}%".format(error_rate.mean()*100))
        if error_rate.mean() < error_threshold:
            print("pass")
            return True
        else:
            print(diff, tensor_golden[golden_mask])
            print("fail")
            return False

def ceil_div(x, y):
    return (x + y - 1) // y

def ceil_align(x, y):
    return ceil_div(x, y) * y

def sizeof(tensor, dtype):
    return tensor.nelement() * dtype

def int8_to_uint8(value):
    if value < 0:
        return value + 256
    else:
        return value

def float32_to_bytes(value):
    num_bytes = 4
    float_bytes = struct.pack('<f', float(value))  # 将float打包成字节，使用小端序
    np_value = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(min(num_bytes, 4)):  # float是4字节，所以最多取4个字节
        np_value[i] = float_bytes[i]
    return np_value

def int_to_bytes(num_bytes, value):
    num_bits = np.int64(num_bytes*8)
    max_value = 2**num_bits - 1
    clip_value = int(np.clip(value, 0, max_value))
    np_value = np.zeros(num_bytes, dtype = np.uint8)
    value_tobytes = clip_value.to_bytes(num_bytes, byteorder = 'little')
    for i in range(num_bytes):
        np_value[i] = value_tobytes[i]
    return np_value

def float32_to_int(value):
    value_tobytes = struct.pack('f', value)
    integer = int.from_bytes(value_tobytes, byteorder = 'little')
    return integer