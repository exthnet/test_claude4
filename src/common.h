#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// BF16 type as uint16_t. GCC has __bf16 since 13.x but we keep portable.
typedef uint16_t bf16_t;

static inline bf16_t fp32_to_bf16(float f) {
    // Round-to-nearest-even.
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t x = v.u;
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        // NaN: keep top bits.
        return (bf16_t)((x >> 16) | 0x40);
    }
    uint32_t rounding_bias = 0x7fff + ((x >> 16) & 1);
    return (bf16_t)((x + rounding_bias) >> 16);
}

static inline float bf16_to_fp32(bf16_t b) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static inline double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Aligned allocation helper.
static inline void *aligned_alloc64(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

// Initialize buffers.
static inline void fill_bf16_random(bf16_t *X, size_t n, unsigned seed) {
    // Use a simple LCG to be deterministic and quick.
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        float v = ((s >> 9) & 0xffff) / 65535.0f - 0.5f;  // [-0.5, 0.5]
        X[i] = fp32_to_bf16(v);
    }
}

static inline void fill_fp32_zero(float *X, size_t n) {
    memset(X, 0, n * sizeof(float));
}

// Naive reference C(MxN) = A(MxK) * B(KxN) using bf16 inputs accumulated in fp32.
// A: row-major MxK, B: row-major KxN, C: row-major MxN.
// Slow; use only for small sizes.
static inline void gemm_reference(int M, int N, int K,
                                  const bf16_t *A, int lda,
                                  const bf16_t *B, int ldb,
                                  float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = bf16_to_fp32(A[(size_t)i * lda + k]);
                float b = bf16_to_fp32(B[(size_t)k * ldb + j]);
                acc += a * b;
            }
            C[(size_t)i * ldc + j] = acc;
        }
    }
}

// Verification: compare two fp32 matrices with relative tolerance.
static inline double max_relative_error(const float *X, const float *Y, size_t n) {
    double maxerr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double a = X[i], b = Y[i];
        double diff = fabs(a - b);
        double denom = fabs(a) + fabs(b) + 1e-30;
        double rel = diff / denom;
        if (rel > maxerr) maxerr = rel;
    }
    return maxerr;
}

static inline double frobenius_relative_error(const float *X, const float *Y, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)X[i] - (double)Y[i];
        num += d * d;
        den += (double)X[i] * (double)X[i];
    }
    if (den == 0.0) return sqrt(num);
    return sqrt(num / den);
}

#endif
