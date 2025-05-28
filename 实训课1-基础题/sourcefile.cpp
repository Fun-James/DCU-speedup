#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <mpi.h>


// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);  // 固定种子以保证结果可重现
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(gen);
    }
    return;
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A[i] - B[i]) > tol) {
            std::cout << "Validation failed at index " << i << ": " 
                      << A[i] << " vs " << B[i] << " (diff: " << std::abs(A[i] - B[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    // 标准三重循环矩阵乘法算法
    // C[i][j] = Σ(A[i][k] * B[k][j]) for k from 0 to M-1
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0.0;  // 初始化为0
            for (int k = 0; k < M; ++k) {
                C[i * P + j] += A[i * M + k] * B[k * P + j];
            }
        }
    }
    return;
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
    std::cout << "matmul_openmp methods..." << std::endl;
    
    // 使用OpenMP并行化最外层循环
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0.0;  // 初始化为0
            for (int k = 0; k < M; ++k) {
                C[i * P + j] += A[i * M + k] * B[k * P + j];
            }
        }
    }
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size = 64) {
    std::cout << "matmul_block_tiling methods..." << std::endl;
    
    // 初始化结果矩阵
    std::fill(C.begin(), C.end(), 0.0);
    
    // 子块并行化矩阵乘法
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                // 计算子块的边界
                int i_end = std::min(ii + block_size, N);
                int j_end = std::min(jj + block_size, P);
                int k_end = std::min(kk + block_size, M);
                
                // 在子块内执行矩阵乘法
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        double sum = 0.0;
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i * M + k] * B[k * P + j];
                        }
                        C[i * P + j] += sum;
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    std::cout << "matmul_mpi methods..." << std::endl;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 分配矩阵内存
    std::vector<double> A, B, C;
    std::vector<double> local_A, local_C;
    
    // 计算每个进程处理的行数
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // 主进程初始化矩阵
    if (rank == 0) {
        A.resize(N * M);
        B.resize(M * P);
        C.resize(N * P, 0.0);
        
        init_matrix(A, N, M);
        init_matrix(B, M, P);
        
        std::cout << "Matrix sizes: A(" << N << "x" << M << ") * B(" << M << "x" << P << ") = C(" << N << "x" << P << ")" << std::endl;
        std::cout << "Using " << size << " MPI processes" << std::endl;
    }
    
    // 分配本地数据
    local_A.resize(local_rows * M);
    local_C.resize(local_rows * P, 0.0);
    B.resize(M * P);  // 所有进程都需要完整的B矩阵
    
    // 广播B矩阵到所有进程
    if (rank == 0) {
        MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // 分发A矩阵的行到各个进程
    std::vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = proc_rows * M;
        displs[i] = offset;
        offset += proc_rows * M;
    }
    
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 计算本地矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < P; ++j) {
            local_C[i * P + j] = 0.0;
            for (int k = 0; k < M; ++k) {
                local_C[i * P + j] += local_A[i * M + k] * B[k * P + j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 收集结果
    std::vector<int> recvcounts(size), recv_displs(size);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = proc_rows * P;
        recv_displs[i] = offset;
        offset += proc_rows * P;
    }
    
    MPI_Gatherv(local_C.data(), local_rows * P, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Process " << rank << " computation time: " << duration.count() << " milliseconds" << std::endl;
        
        // 验证结果
        std::vector<double> C_ref(N * P, 0.0);
        matmul_baseline(A, B, C_ref, N, M, P);
        std::cout << "[MPI] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else {
        std::cout << "Process " << rank << " computation time: " << duration.count() << " milliseconds" << std::endl;
    }
}

// 方式4: 其他方式 ：OpenMP+MPI
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    std::cout << "matmul_other (MPI + OpenMP Block Tiling) methods..." << std::endl;

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // --- MPI Data Distribution ---
    int rows_per_proc = N / mpi_size;
    int remainder_rows = N % mpi_size;
    // Calculate number of rows this process handles for A and C
    int local_N = rows_per_proc + (rank < remainder_rows ? 1 : 0); 
    
    // Calculate displacement for each process in the original A and C matrices
    std::vector<int> sendcounts_A(mpi_size);
    std::vector<int> displs_A(mpi_size);
    std::vector<int> recvcounts_C(mpi_size);
    std::vector<int> displs_C(mpi_size);

    int current_displ_A = 0;
    int current_displ_C = 0;
    for (int i = 0; i < mpi_size; ++i) {
        int rows_for_proc_i = rows_per_proc + (i < remainder_rows ? 1 : 0);
        sendcounts_A[i] = rows_for_proc_i * M; // elements in A
        displs_A[i] = current_displ_A;
        current_displ_A += sendcounts_A[i];

        recvcounts_C[i] = rows_for_proc_i * P; // elements in C
        displs_C[i] = current_displ_C;
        current_displ_C += recvcounts_C[i];
    }

    std::vector<double> local_A(local_N * M);
    std::vector<double> local_C(local_N * P, 0.0);
    
    // Scatter A from rank 0 to local_A on all processes
    // Only rank 0 has valid A data, others should pass nullptr
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                 local_A.data(), local_N * M, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast B to all processes.
    // B is const&. MPI_Bcast needs a non-const buffer for receiving.
    // So, all processes prepare a buffer. Rank 0 copies B into it.
    std::vector<double> B_broadcast(M * P);
    if (rank == 0) {
        std::copy(B.begin(), B.end(), B_broadcast.begin());
    }
    MPI_Bcast(B_broadcast.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Local Computation (OpenMP Sub-block Tiling) ---
    // Each MPI process computes its part of C (local_C) using local_A and B_broadcast.
    // The OpenMP parallelism is within this block.
    int block_size = 64; // Define your desired block size
    std::fill(local_C.begin(), local_C.end(), 0.0); // Initialize local_C for accumulation

    // Parallelize the outer two loops for better load distribution
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < local_N; ii += block_size) { // Row blocks of local_C
        for (int jj = 0; jj < P; jj += block_size) {   // Column blocks of local_C / B_broadcast
            // Sequential k-loop to avoid race conditions on local_C[i*P+j]
            for (int kk = 0; kk < M; kk += block_size) { // K-dimension blocks (matrix M)
                
                int i_end = std::min(ii + block_size, local_N);
                int j_end = std::min(jj + block_size, P);
                int k_end = std::min(kk + block_size, M);

                for (int i = ii; i < i_end; ++i) {        // Current row in local_A / local_C
                    for (int j = jj; j < j_end; ++j) {    // Current column in B_broadcast / local_C
                        double sum_for_k_block = 0.0;
                        for (int k = kk; k < k_end; ++k) { // Innermost loop for dot product part
                            sum_for_k_block += local_A[i * M + k] * B_broadcast[k * P + j];
                        }
                        local_C[i * P + j] += sum_for_k_block; // Accumulate result
                    }
                }
            }
        }
    }

    // --- MPI Data Aggregation ---
    // Gather local_C from all processes to C on rank 0
    // Only rank 0 needs to provide receive buffer
    MPI_Gatherv(local_C.data(), local_N * P, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts_C.data(), displs_C.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    // Rank 0 now has the complete matrix C.
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    // MPI模式处理
    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        matmul_mpi(N, M, P);
        MPI_Finalize();
        return 0;
    }
    
    // OpenMP+MPI混合模式处理
    if (mode == "other") {
        MPI_Init(&argc, &argv);
        
        int rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        std::vector<double> A, B, C, C_ref;
        
        // 只有主进程初始化完整矩阵
        if (rank == 0) {
            A.resize(N * M);
            B.resize(M * P);
            C.resize(N * P, 0);
            C_ref.resize(N * P, 0);
            
            init_matrix(A, N, M);
            init_matrix(B, M, P);
            
            std::cout << "Pre-calculating reference matrix C_ref for validation purposes..." << std::endl;
            auto ref_calc_start_time = std::chrono::high_resolution_clock::now();
            matmul_baseline(A, B, C_ref, N, M, P);
            auto ref_calc_end_time = std::chrono::high_resolution_clock::now();
            auto ref_calc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ref_calc_end_time - ref_calc_start_time);
            std::cout << "C_ref pre-calculation time: " << ref_calc_duration.count() << " milliseconds." << std::endl << std::endl;
        } else {
            // 非主进程不需要分配完整的A和B矩阵，只需要为结果分配空间
            // A和B会在matmul_other函数内部通过MPI分发和广播
            C.resize(N * P, 0);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (rank == 0) {
            std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
            std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
        }
        
        MPI_Finalize();
        return 0;
    }

    // 非MPI模式的处理保持不变
    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);

    std::cout << "Pre-calculating reference matrix C_ref for validation purposes..." << std::endl;
    auto ref_calc_start_time = std::chrono::high_resolution_clock::now();
    matmul_baseline(A, B, C_ref, N, M, P);
    auto ref_calc_end_time = std::chrono::high_resolution_clock::now();
    auto ref_calc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ref_calc_end_time - ref_calc_start_time);
    std::cout << "C_ref pre-calculation time: " << ref_calc_duration.count() << " milliseconds." << std::endl << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    if (mode == "baseline") {
        std::cout << "[Baseline] Computing matrix multiplication (this is the timed section)..." << std::endl;
        std::cout << "Matrix sizes: A(" << N << "x" << M << ") * B(" << M << "x" << P << ") = C(" << N << "x" << P << ")" << std::endl;
        matmul_baseline(A, B, C, N, M, P); 
        std::cout << "[Baseline] Done. Result validation (C vs C_ref): " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "openmp") {
        matmul_openmp(A, B, C, N, M, P);
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else if (mode == "block") {
        matmul_block_tiling(A, B, C, N, M, P);
        std::cout << "[Block Parallel] Valid: " << validate(C, C_ref, N, P) << std::endl;
    } else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi|other]" << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    
    return 0;
}
