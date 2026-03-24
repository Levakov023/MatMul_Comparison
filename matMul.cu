#include <iostream>
#include <chrono>

__global__ void matMul(float* A, float* B, float* C, int a_R, int b_C, int K) {

    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    const auto col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < a_R && col < b_C) {
        float cValue = 0;

        for ( auto counter{0}; counter < K ; counter++ ) {
            cValue += A[row * K + counter] * B[counter * b_C + col];
        }
        C[row * b_C + col] = cValue;
    }
}


int main() {
    //initialize random matrix shapes for matmul,  note : M x K * K x N is needed .
    int K = 1024;
    int a_rows = 718;
    int b_columns = 556;

    auto* h_A = new float[a_rows*K];
    auto* h_B = new float[b_columns*K];

    auto* h_C = new float[a_rows * b_columns];

    // Here goes random matrix making
    for ( int counter{0}; counter < a_rows * K ; counter++ ) {
        h_A[counter] = static_cast<float>(rand()) / RAND_MAX;
    }

    for ( int counter{0}; counter < b_columns * K ; counter++ ) {
        h_B[counter] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_rows * K * sizeof(float));
    cudaMalloc(&d_B, b_columns * K * sizeof(float));
    cudaMalloc(&d_C, a_rows * b_columns * sizeof(float));

    // Here goes cudaMemcpy() stuff

    cudaMemcpy(d_A, h_A, a_rows * K * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_B, h_B, b_columns * K * sizeof(float), cudaMemcpyHostToDevice );

    dim3 blockSize(16, 16);
    dim3 gridSize((a_rows +15) / 16, ( b_columns + 15 )/16);

    /*
     * profiling for CPU vs naive GPU vs tiled GPU approach.
     */
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cudaEvent_t cudaStart, cudaStop;
    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    cudaEventRecord(cudaStart);

    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, a_rows, b_columns, K);
    cudaDeviceSynchronize();

    cudaEventRecord(cudaStop);
    cudaEventSynchronize(cudaStop);

    auto cpuEnd = std::chrono::high_resolution_clock::now();


    float ms = 0;
    cudaEventElapsedTime(&ms, cudaStart, cudaStop);
    std::cout << "GPU kernel : " << ms << "ms\n";

    std::chrono::duration<double> diff = cpuEnd - cpuStart;
    std::cout << "Wall clock: " << diff.count() * 1000 << "ms\n";

    // Print out supposed results.
    float expected = 0;
    for (int k = 0; k < K; k++)
        expected += h_A[k] * h_B[k * b_columns];
    std::cout << "expected C[0][0] = " << expected << "\n";


    cudaMemcpy(h_C, d_C, a_rows * b_columns * sizeof(float), cudaMemcpyDeviceToHost);


    // print out kernel matMul results.
    for (int i = 0; i < 5; i++)
        std::cout << "C[" << i << "] = " << h_C[i] << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
