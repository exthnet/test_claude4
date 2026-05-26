/* Benchmark driver for bf16 x bf16 -> fp32 GEMM */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "amx_util.h"

void naive_gemm(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc);
void avx512bf16_gemm(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc);
void amx_gemm_bf16(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc);
void mkl_gemm_bf16(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc);
void mkl_init_single_thread(void);

typedef void (*gemm_fn_t)(int, int, int, const bf16_t*, int, const bf16_t*, int, float*, int);

struct method {
    const char *name;
    gemm_fn_t fn;
    int big_ok;  /* 1 if can handle big sizes in reasonable time */
};

static struct method methods[] = {
    { "naive",   naive_gemm,        0 },
    { "avx512",  avx512bf16_gemm,   1 },
    { "amx",     amx_gemm_bf16,     1 },
    { "mkl",     mkl_gemm_bf16,     1 },
    { NULL, NULL, 0 }
};

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <method> <M> <N> <K> [iters] [warmups] [verify=0|1]\n"
        "  method: naive | avx512 | amx | mkl | all\n", prog);
}

static int find_method(const char *name) {
    for (int i = 0; methods[i].name; i++) if (strcmp(methods[i].name, name) == 0) return i;
    return -1;
}

static void bench_one(const char *name, gemm_fn_t fn,
                      int M, int N, int K,
                      const bf16_t *A, const bf16_t *B,
                      float *C, float *C_ref,
                      int iters, int warmups, int verify)
{
    /* Warmups */
    for (int w = 0; w < warmups; w++) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        fn(M, N, K, A, K, B, N, C, N);
    }

    /* Timed iterations */
    double tmin = 1e30, tmax = 0.0, tsum = 0.0;
    for (int it = 0; it < iters; it++) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        double t0 = wall_time();
        fn(M, N, K, A, K, B, N, C, N);
        double t1 = wall_time();
        double dt = t1 - t0;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
        tsum += dt;
    }
    double tavg = tsum / iters;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_avg = flops / tavg / 1e9;
    double gflops_best = flops / tmin / 1e9;

    printf("METHOD=%-7s M=%5d N=%5d K=%5d  iters=%d  t_avg=%.4fs  t_min=%.4fs  GFLOPS(avg)=%.2f  GFLOPS(best)=%.2f",
           name, M, N, K, iters, tavg, tmin, gflops_avg, gflops_best);

    if (verify && C_ref) {
        double rel = frobenius_relative_error(C_ref, C, (size_t)M * N);
        printf("  rel_err=%.3e", rel);
    }
    printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }
    const char *meth = argv[1];
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);
    int iters   = (argc > 5) ? atoi(argv[5]) : 3;
    int warmups = (argc > 6) ? atoi(argv[6]) : 1;
    int verify  = (argc > 7) ? atoi(argv[7]) : 0;

    /* AMX permission */
    if (amx_request_permission() != 0) {
        fprintf(stderr, "Failed to request AMX permission.\n");
        return 1;
    }

    /* MKL: force single-thread */
    mkl_init_single_thread();

    /* Allocate buffers */
    bf16_t *A = aligned_alloc64((size_t)M * K * sizeof(bf16_t));
    bf16_t *B = aligned_alloc64((size_t)K * N * sizeof(bf16_t));
    float  *C = aligned_alloc64((size_t)M * N * sizeof(float));
    float  *C_ref = NULL;
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n"); return 2;
    }

    fill_bf16_random(A, (size_t)M * K, 12345);
    fill_bf16_random(B, (size_t)K * N, 67890);

    if (verify) {
        C_ref = aligned_alloc64((size_t)M * N * sizeof(float));
        if (!C_ref) { fprintf(stderr, "alloc C_ref failed\n"); return 3; }
        if (M <= 1024 && N <= 1024 && K <= 1024) {
            /* Use naive as ground truth for small sizes. */
            naive_gemm(M, N, K, A, K, B, N, C_ref, N);
        } else {
            /* Use MKL as ground truth for large sizes (will be available). */
            mkl_gemm_bf16(M, N, K, A, K, B, N, C_ref, N);
        }
    }

    if (strcmp(meth, "all") == 0) {
        for (int i = 0; methods[i].name; i++) {
            if (!methods[i].big_ok && ((size_t)M * N * K > (1u<<28))) continue;  /* skip naive for big */
            bench_one(methods[i].name, methods[i].fn, M, N, K, A, B, C, C_ref, iters, warmups, verify);
        }
    } else {
        int idx = find_method(meth);
        if (idx < 0) { usage(argv[0]); return 1; }
        bench_one(methods[idx].name, methods[idx].fn, M, N, K, A, B, C, C_ref, iters, warmups, verify);
    }

    free(A); free(B); free(C); if (C_ref) free(C_ref);
    return 0;
}
