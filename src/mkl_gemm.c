/* MKL cblas_gemm_bf16bf16f32 reference. */
#include "common.h"

#ifdef USE_MKL
#include <mkl.h>

void mkl_gemm_bf16(int M, int N, int K,
                   const bf16_t *A, int lda,
                   const bf16_t *B, int ldb,
                   float *C, int ldc)
{
    float alpha = 1.0f, beta = 0.0f;
    /* MKL's MKL_BF16 is an unsigned short with the same bit layout we use. */
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K,
                           alpha, (const MKL_BF16*)A, lda,
                                  (const MKL_BF16*)B, ldb,
                           beta,  C, ldc);
}

void mkl_init_single_thread(void) {
    mkl_set_num_threads(1);
    /* Disable dynamic to keep it 1 thread. */
    mkl_set_dynamic(0);
}

#else  /* USE_MKL not defined */

#include <stdio.h>
void mkl_gemm_bf16(int M, int N, int K,
                   const bf16_t *A, int lda,
                   const bf16_t *B, int ldb,
                   float *C, int ldc)
{
    (void)M; (void)N; (void)K; (void)A; (void)lda; (void)B; (void)ldb; (void)C; (void)ldc;
    fprintf(stderr, "MKL not compiled in.\n");
}
void mkl_init_single_thread(void) {}

#endif
