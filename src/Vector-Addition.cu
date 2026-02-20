#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>

/*
Steps for execution
1. Set CUDA path in ~/.bashrc file
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

2. Source bash file using the command: source .bashrc
3. Compile the program file using the command: nvcc Vector-Addition.cu -o Vector-Addition
4. Execute the program using the command: ./Vector-Addition and view the output.
*/

// CUDA Kernel for Vector Addition
__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Sequential CPU Implementation
void vectorAddCPU(const int* a, const int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void randomInit(int* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (rand() % 21) - 10;
    }
}

struct Results {
    float configID;
    int vectorSize;
    int blocks;
    int threadsPerBlock;
    float inclusiveTime; 
    float kernelOnlyTime;
};

// --- Sequential Flow Runner ---
void runSequentialExperiment(int N, std::vector<Results>& allResults, float configId) {
    size_t size = N * sizeof(int);

    auto startTotal = std::chrono::high_resolution_clock::now();
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    randomInit(h_a, N);
    randomInit(h_b, N);

    auto start = std::chrono::high_resolution_clock::now();
    
    vectorAddCPU(h_a, h_b, h_c, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto endTotal = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> totalMs = endTotal - startTotal;
    std::chrono::duration<float, std::milli> mathMs = end - start;

    // Print Stats (Blocks/Threads marked as 0 for CPU)
    std::cout << std::setw(10) << N << " | " 
              << std::setw(6) << 0 << " | " 
              << std::setw(8) << 0 << " | " 
              << std::fixed << std::setprecision(4) << std::setw(12) << totalMs.count() << " | " 
              << std::setw(12) << mathMs.count() << std::endl;

    allResults.push_back({configId, N, 0, 0, totalMs.count(), mathMs.count()});

    free(h_a); free(h_b); free(h_c);
}

// --- GPU Flow Runner ---
void runExperiment(int N, int blocks, int threads, std::vector<Results>& allResults, float configId) {
    size_t size = N * sizeof(int);
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    randomInit(h_a, N);
    randomInit(h_b, N);

    cudaEvent_t startK, stopK;
    cudaEventCreate(&startK);
    cudaEventCreate(&stopK);
    
    auto startTotal = std::chrono::high_resolution_clock::now();

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEventRecord(startK);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    // Check for launch errors
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    auto endTotal = std::chrono::high_resolution_clock::now();
    
    float kernelMs = 0;
    cudaEventElapsedTime(&kernelMs, startK, stopK);
    std::chrono::duration<float, std::milli> totalMs = endTotal - startTotal;

    // Print Stats to Terminal
    std::cout << std::setw(10) << N << " | " 
              << std::setw(6) << blocks << " | " 
              << std::setw(8) << threads << " | " 
              << std::fixed << std::setprecision(4) << std::setw(12) << totalMs.count() << " | " 
              << std::setw(12) << kernelMs << std::endl;

    allResults.push_back({configId, N, blocks, threads, totalMs.count(), kernelMs});

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    cudaEventDestroy(startK); cudaEventDestroy(stopK);
}

void printHeader(const char* title) {
    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << " FLOW: " << title << "\n";
    std::cout << std::string(65, '-') << "\n";
    std::cout << std::setw(10) << "Size" << " | " << std::setw(6) << "Blocks" << " | " 
              << std::setw(8) << "Threads" << " | " << std::setw(12) << "Incl. ms" << " | " 
              << std::setw(12) << "Kernel ms" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
}

int main() {
    std::vector<Results> allResults;
    srand(time(0));

    // Warm-up to initialize CUDA context
    int *dummy; cudaMalloc(&dummy, 4); cudaFree(dummy);

    // --- NEW FLOW 0: Sequential CPU Execution ---
    printHeader("0.0 - Sequential CPU Baseline");
    for (int p = 3; p <= 20; p++) {
        int N = pow(2, p);
        runSequentialExperiment(N, allResults, 0);
    }

    // --- FLOW 1.1: Fixed 1 Block, Varying Threads ---
    printHeader("1.1 - Fixed 1 Block, Varying Threads");
    for (int p = 3; p <= 20; p++) {
        int N = pow(2, p);
        int threads = min(N, 1024);
        runExperiment(N, 1, threads, allResults, 1.1);
    }

    // --- FLOW 1.2: Fixed 1024 Threads, Varying Blocks ---
    printHeader("1.2 - Fixed 1024 Threads, Varying Blocks (1 to 128)");
    for (int p = 3; p <= 20; p++) {
        int N = pow(2, p);
        int blocks = (N + 1023) / 1024;
        blocks = min(blocks, 128);
        runExperiment(N, blocks, 1024, allResults, 1.2);
    }

    // --- FLOW 1.3: Varying Threads and Blocks ---
    printHeader("1.3 - Varying Both (Adaptive Grid)");
    for (int p = 3; p <= 20; p++) {
        int N = pow(2, p);
        int threads = min(N, 1024); 
        int blocks = (N + threads - 1) / threads;
        runExperiment(N, blocks, threads, allResults, 1.3);
    }

    // --- FLOW 2.1: Stress Test - Fixed 1 Block, Fixed 1 Thread ---
    printHeader("2.1 - Stress Test - Fixed 1 Block, Fixed 1 Thread");
    for (int p = 3; p <= 20; p++) {
        int N = pow(2, p);
        runExperiment(N, 1, 1, allResults, 2.1);
    }

    // --- FLOW 2.2: Stress Test - Fixed 1 Block, Exceeding Maximum threads ---
    printHeader("2.2 - Fixed 1 Block, Exceeding Maximum threads");
    for (int p = 10; p <= 20; p++) {
        int N = pow(2, p);
        int threads = N;
        runExperiment(N, 1, threads, allResults, 2.2);
    }

    // Write to CSV
    std::ofstream file("cuda_results.csv");
    file << "ConfigID,VectorSize,Blocks,ThreadsPerBlock,InclusiveTime_ms,KernelTime_ms\n";
    for (const auto& r : allResults) {
        file << r.configID << "," << r.vectorSize << "," << r.blocks << "," << r.threadsPerBlock << "," 
             << r.inclusiveTime << "," << r.kernelOnlyTime << "\n";
    }
    file.close();

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "Data collection complete. File saved: cuda_results.csv" << std::endl;

    return 0;
}