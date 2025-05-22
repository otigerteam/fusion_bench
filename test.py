import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------
# Three separate kernels: GEMM, Bias, and ReLU
# -------------------------------------------------------------
original_cuda = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

constexpr int TILE = 16;

// GEMM Kernel + Wrapper
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float Asub[TILE][TILE], Bsub[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aRow = row, aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y, bCol = col;
        Asub[threadIdx.y][threadIdx.x] =
            (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] =
            (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K
    );
    return C;
}

// Bias Kernel + Wrapper
__global__ void add_bias_kernel(float* C, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] += bias[col];
    }
}

torch::Tensor add_bias(torch::Tensor C, torch::Tensor bias) {
    auto M = C.size(0), N = C.size(1);
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    add_bias_kernel<<<grid, block>>>(
        C.data_ptr<float>(), bias.data_ptr<float>(),
        M, N
    );
    return C;
}

// ReLU Kernel + Wrapper
__global__ void relu_kernel(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float v = C[row * N + col];
        C[row * N + col] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor relu(torch::Tensor C) {
    auto M = C.size(0), N = C.size(1);
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    relu_kernel<<<grid, block>>>(
        C.data_ptr<float>(), M, N
    );
    return C;
}
"""

original_cpp = """
 torch::Tensor gemm(torch::Tensor, torch::Tensor);
 torch::Tensor add_bias(torch::Tensor, torch::Tensor);
 torch::Tensor relu(torch::Tensor);
"""

original_ext = load_inline(
    name="original_kernels",
    cpp_sources=[original_cpp],
    cuda_sources=[original_cuda],
    functions=["gemm", "add_bias", "relu"],
    verbose=True,
    extra_cflags=['-I/opt/rocm/include'],
    extra_cuda_cflags=['-I/opt/rocm/include'],
    extra_ldflags=['-L/opt/rocm/lib', "-lamdhip64"],
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = original_ext.gemm
        self.add_bias = original_ext.add_bias
        self.relu = original_ext.relu

    def forward(self, A, B, bias):
        C = self.gemm(A, B)
        C = self.add_bias(C, bias)
        C = self.relu(C)
        return C
 
# M, K, N = 64, 64, 64
M, K, N = 1024, 1024, 1024

def get_inputs():
    # Generate random input matrices A (M×K), B (K×N), and bias vector (N)
    A    = torch.randn(M, K).cuda()
    B    = torch.randn(K, N).cuda()
    bias = torch.randn(N).cuda()
    return [A, B, bias]

def get_init_inputs():
    # No extra init args needed for OriginalModel
    return []
