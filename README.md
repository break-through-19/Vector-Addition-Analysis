# Vector-Addition-Analysis
Varying sizes of vector addition is compared across CPU and GPU (varying blocks size and thread count)

## 1. Variants of the Configurations

The project evaluates vector addition across multiple execution flows to analyze performance scaling:

- **Flow 0.0 (Sequential CPU Baseline):** Runs vector addition sequentially on the CPU for performance comparison against the GPU.
- **Flow 1.1 (Fixed 1 Block, Varying Threads):** Uses a single block while scaling threads up to the hardware limit (1024). Explores intra-block parallelism.
- **Flow 1.2 (Fixed 1024 Threads, Varying Blocks):** Uses maximum threads per block (1024) while scaling the grid dimension (1 to 128 blocks). Explores inter-block parallelism.
- **Flow 1.3 (Adaptive Grid - Varying Both):** Dynamically calculates the required blocks based on the data size (`(N + threads - 1) / threads`). This represents the optimal and standard approach for arbitrarily sized arrays.
- **Flow 2.1 (Stress Test - 1 Block, 1 Thread):** Represents pure sequential execution on the GPU to measure individual thread performance.
- **Flow 2.2 (Stress Test - Exceeding Maximum Threads):** Intentionally exceeds the 1024 threads-per-block limit to observe launch failures and validate hardware thread constraints.

## 2. Highlights and Key Findings

Based on the collected performance metrics (`cuda_results.csv`):
- **CPU vs GPU Overhead for Small Data:** For small vector sizes (e.g., N=8), the CPU sequentially outperforms the GPU in total execution time due to the high latency of memory transfers (PCIe) and kernel launch overhead.
- **Massive GPU Speedup for Large Data:** For large vector sizes (e.g., N=1,048,576), the CPU requires ~21.5 ms. In contrast, an adaptively scaled GPU configuration (Flow 1.3) requires ~1.3 ms total time and only ~0.027 ms for the pure kernel execution itself—demonstrating massive parallel compute capabilities.
- **Block Limitations Tested:** Flow 2.2 confirms the hardware limitations on modern NVIDIA GPUs. Attempting to launch a block with more than 1024 threads causes the kernel to fail its launch sequence, resulting in 0 computation time (aborted).
- **Parallelism Dominates Clock Speed:** Flow 2.1 (1 Block, 1 Thread) performs poorly on the GPU computationally compared to multithreaded flows and CPU execution, showcasing that GPU architecture relies heavily on high concurrency and throughput rather than single-core frequency.

## 3. Tech Stack

- **Language:** C++, CUDA C++
- **Framework:** CUDA Toolkit (NVIDIA GPU Computing Platform)
- **Standard Libraries:** `<iostream>`, `<vector>`, `<fstream>`, `<chrono>`
- **Compiler:** `nvcc` (NVIDIA CUDA Compiler)

## 4. Technical Definition of Fundamental Terms

- **CUDA (Compute Unified Device Architecture):** A parallel computing platform and programming model developed by NVIDIA that allows developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing.
- **Kernel Function:** A C++ function designed to be executed on the GPU device. When called, it is executed $N$ times in parallel by $N$ different CUDA threads, as opposed to only once like regular C++ functions. It is defined using the `__global__` declaration specifier.
- **Thread:** The smallest unit of execution on the GPU. Thousands of independent threads can be executed concurrently to process data arrays efficiently.
- **Block (Thread Block):** A grouping of threads that execute the same kernel and can cooperate with each other via shared memory and synchronization barriers. Hardware architecture currently limits blocks to a maximum of 1024 threads.
- **Grid:** An array of thread blocks that execute the same kernel, allowing the processing to scale efficiently across the GPU's available streaming multiprocessors (SMs).
