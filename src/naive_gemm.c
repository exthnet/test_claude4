/* Triple-loop naive GEMM. Only useful for small sizes. */
#include "common.h"

void naive_gemm(int M, int N, int K,
                const bf16_t *A, int lda,
                const bf16_t *B, int ldb,
                float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += bf16_to_fp32(A[(size_t)i * lda + k]) * bf16_to_fp32(B[(size_t)k * ldb + j]);
            }
            C[(size_t)i * ldc + j] = acc;
        }
    }
}
