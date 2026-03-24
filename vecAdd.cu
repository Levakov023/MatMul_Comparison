#include <iostream>



__global__ void vecAdd(float* A, float* B, float* C, int N) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

/*
* dim3 blockSize(16, 16);
* dim3 gridsize(ceil(b_R/16), ceil(a_C/16));
 */

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    auto* h_A = new float[N];
    auto* h_B = new float[N];
    auto* h_C = new float[N];

    for (int i = 0; i < N ; i++ ) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;

    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    /*
     * because integer division we need to add 255 (threadsperblock -1)
     * at N =1000, 1000/256 -> 3.9, truncates to 3.0 (not enough)
     *  N = 1255 /256, -> 4.9, truncates to 4.0, enough warps
     */

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err) << '\n';

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "h_C[0]: " << h_C[0] << '\n';

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}