from semantic import *
from common import *
from utils import *
from ops.matmul import *
from ops.activation import *

class BlockTestMatmul(unittest.TestCase):
    def setUp(self):
        print("="*100)
        print("BlockTestMatmul begin")

    def test_matmul_tile_twice(self):
        print("test_matmul_tile_twice")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.matmul(left, right.permute(1, 0)) + bias.unsqueeze(0)
            result_mn = op_matmul_tile_twice(left, right, bias) ### 默认情况下左矩阵的layout是MK，右矩阵的layout是NK
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_matmul_tile_once(self):
        print("test_matmul_tile_once")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.matmul(left, right.permute(1, 0)) + bias.unsqueeze(0)
            result_mn = op_matmul_tile_once(left, right, bias)  ### 默认情况下左矩阵的layout是MK，右矩阵的layout是NK
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_matmul_transpose(self):
        print("test_matmul_transpose")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.matmul(left, right.permute(1, 0)) + bias.unsqueeze(0)
            result_mn = op_matmul_transpose(left, right.permute(1, 0), bias) ### 这行代码非常关键，input右矩阵的layout是KN
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def tearDown(self):
        print("BlockTestMatmul end")
        print("="*100)

class BlockTestActivation(unittest.TestCase):
    def setUp(self):
        print("="*100)
        print("BlockTestActivation begin")

    def test_softmax(self):
        print("test_softmax")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.softmax(left, dim=-1)
            result_mn = op_softmax(left)
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_layernorm(self):
        print("test_layernorm")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.layer_norm(left, dim=-1)
            result_mn = op_layernorm(left)
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_rmsnorm(self):
        print("test_rmsnorm")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.rms_norm(left, dim=-1)
            result_mn = op_rmsnorm(left)
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_sigmoid(self):
        print("test_sigmoid")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.sigmoid(left)
            result_mn = op_sigmoid(left)
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def test_silu(self):
        print("test_silu")
        for i in range(3):
            matmul_params = generate_matmul_params()
            left, right, bias = generate_matmul_tensor(matmul_params)
            golden_mn = torch.silu(left)
            result_mn = op_silu(left)
            test_pass = compare(result_mn, golden_mn)
            self.assertEqual(test_pass, True)

    def tearDown(self):
        print("BlockTestActivation end")
        print("="*100)

suite = unittest.TestSuite()
if __name__ == '__main__':
    unittest.main()