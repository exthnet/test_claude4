/*
 * AVX-512 BF16 (_mm512_dpbf16_ps) blocked GEMM as a baseline/comparison.
 * Same packing convention as the AMX path (B in VNNI, A packed in 16-row chunks of kc bf16).
 * Micro-kernel: 8 M rows x 32 N cols fp32 output (16 zmm output regs).
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#ifndef AVX_MC
#define AVX_MC 128
#endif
#ifndef AVX_NC
#define AVX_NC 1024
#endif
#ifndef AVX_KC
#define AVX_KC 512
#endif

#define AVX_MR 8
#define AVX_NR 32
#define AVX_KSTEP 2

static bf16_t *g_Apack = NULL;
static bf16_t *g_Bpack = NULL;
static int g_Apack_cap = 0, g_Bpack_cap = 0;

static void ensure_buffers(int mc, int nc, int kc) {
    size_t an = (size_t)mc * kc;
    size_t bn = (size_t)kc * nc;
    if ((int)an > g_Apack_cap) { if (g_Apack) free(g_Apack); g_Apack = aligned_alloc64(an * 2 + 128); g_Apack_cap = (int)an; }
    if ((int)bn > g_Bpack_cap) { if (g_Bpack) free(g_Bpack); g_Bpack = aligned_alloc64(bn * 2 + 128); g_Bpack_cap = (int)bn; }
}

/* Pack A in 8-row chunks; each chunk has 8 rows of kc bf16 stored contiguously. */
static void pack_A(const bf16_t *A, int lda, int mc, int kc, bf16_t *Apack) {
    int nc8 = mc / AVX_MR;
    for (int c = 0; c < nc8; c++) {
        const bf16_t *src = A + (size_t)(c * AVX_MR) * lda;
        bf16_t *dst = Apack + (size_t)c * AVX_MR * kc;
        for (int r = 0; r < AVX_MR; r++) {
            memcpy(dst + (size_t)r * kc, src + (size_t)r * lda, (size_t)kc * sizeof(bf16_t));
        }
    }
}

/* Same VNNI pack as AMX path: (kc/2) rows of (nc*2) bf16. */
static void pack_B(const bf16_t *B, int ldb, int kc, int nc, bf16_t *Bpack) {
    int kp = kc / 2;
    for (int p = 0; p < kp; p++) {
        const bf16_t *r0 = B + (size_t)(2 * p) * ldb;
        const bf16_t *r1 = B + (size_t)(2 * p + 1) * ldb;
        bf16_t *d = Bpack + (size_t)p * (nc * 2);
        for (int n = 0; n < nc; n++) {
            d[2 * n + 0] = r0[n];
            d[2 * n + 1] = r1[n];
        }
    }
}

/* 8x32 micro-kernel using _mm512_dpbf16_ps. */
static inline void micro_8x32(
    const bf16_t *Ap,   /* 8 rows x kc bf16, contiguous */
    const bf16_t *Bp,   /* VNNI Bpack: (kc/2) rows of (nc*2) bf16, stride ldb_bytes */
    size_t ldb_bytes,
    float *C, size_t ldc_elems,
    int kc,
    int zero_init)
{
    __m512 c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1;
    __m512 c4_0, c4_1, c5_0, c5_1, c6_0, c6_1, c7_0, c7_1;
    if (zero_init) {
        c0_0 = _mm512_setzero_ps(); c0_1 = _mm512_setzero_ps();
        c1_0 = _mm512_setzero_ps(); c1_1 = _mm512_setzero_ps();
        c2_0 = _mm512_setzero_ps(); c2_1 = _mm512_setzero_ps();
        c3_0 = _mm512_setzero_ps(); c3_1 = _mm512_setzero_ps();
        c4_0 = _mm512_setzero_ps(); c4_1 = _mm512_setzero_ps();
        c5_0 = _mm512_setzero_ps(); c5_1 = _mm512_setzero_ps();
        c6_0 = _mm512_setzero_ps(); c6_1 = _mm512_setzero_ps();
        c7_0 = _mm512_setzero_ps(); c7_1 = _mm512_setzero_ps();
    } else {
        c0_0 = _mm512_loadu_ps(C + 0*ldc_elems + 0); c0_1 = _mm512_loadu_ps(C + 0*ldc_elems + 16);
        c1_0 = _mm512_loadu_ps(C + 1*ldc_elems + 0); c1_1 = _mm512_loadu_ps(C + 1*ldc_elems + 16);
        c2_0 = _mm512_loadu_ps(C + 2*ldc_elems + 0); c2_1 = _mm512_loadu_ps(C + 2*ldc_elems + 16);
        c3_0 = _mm512_loadu_ps(C + 3*ldc_elems + 0); c3_1 = _mm512_loadu_ps(C + 3*ldc_elems + 16);
        c4_0 = _mm512_loadu_ps(C + 4*ldc_elems + 0); c4_1 = _mm512_loadu_ps(C + 4*ldc_elems + 16);
        c5_0 = _mm512_loadu_ps(C + 5*ldc_elems + 0); c5_1 = _mm512_loadu_ps(C + 5*ldc_elems + 16);
        c6_0 = _mm512_loadu_ps(C + 6*ldc_elems + 0); c6_1 = _mm512_loadu_ps(C + 6*ldc_elems + 16);
        c7_0 = _mm512_loadu_ps(C + 7*ldc_elems + 0); c7_1 = _mm512_loadu_ps(C + 7*ldc_elems + 16);
    }

    int kp = kc / 2;
    for (int p = 0; p < kp; p++) {
        const bf16_t *Brow = (const bf16_t*)((const char*)Bp + (size_t)p * ldb_bytes);
        __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(Brow + 0));
        __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(Brow + 32));

        /* For each row, load a 32-bit (2 bf16) pair from A and broadcast across the lanes. */
        #define DO_ROW(rowidx, cvar0, cvar1) do { \
            const uint32_t *Aptr = (const uint32_t*)(Ap + (size_t)(rowidx) * kc + 2 * p); \
            __m512bh a = (__m512bh)_mm512_set1_epi32((int)*Aptr); \
            cvar0 = _mm512_dpbf16_ps(cvar0, a, b0); \
            cvar1 = _mm512_dpbf16_ps(cvar1, a, b1); \
        } while (0)

        DO_ROW(0, c0_0, c0_1);
        DO_ROW(1, c1_0, c1_1);
        DO_ROW(2, c2_0, c2_1);
        DO_ROW(3, c3_0, c3_1);
        DO_ROW(4, c4_0, c4_1);
        DO_ROW(5, c5_0, c5_1);
        DO_ROW(6, c6_0, c6_1);
        DO_ROW(7, c7_0, c7_1);
        #undef DO_ROW
    }

    _mm512_storeu_ps(C + 0*ldc_elems + 0,  c0_0); _mm512_storeu_ps(C + 0*ldc_elems + 16, c0_1);
    _mm512_storeu_ps(C + 1*ldc_elems + 0,  c1_0); _mm512_storeu_ps(C + 1*ldc_elems + 16, c1_1);
    _mm512_storeu_ps(C + 2*ldc_elems + 0,  c2_0); _mm512_storeu_ps(C + 2*ldc_elems + 16, c2_1);
    _mm512_storeu_ps(C + 3*ldc_elems + 0,  c3_0); _mm512_storeu_ps(C + 3*ldc_elems + 16, c3_1);
    _mm512_storeu_ps(C + 4*ldc_elems + 0,  c4_0); _mm512_storeu_ps(C + 4*ldc_elems + 16, c4_1);
    _mm512_storeu_ps(C + 5*ldc_elems + 0,  c5_0); _mm512_storeu_ps(C + 5*ldc_elems + 16, c5_1);
    _mm512_storeu_ps(C + 6*ldc_elems + 0,  c6_0); _mm512_storeu_ps(C + 6*ldc_elems + 16, c6_1);
    _mm512_storeu_ps(C + 7*ldc_elems + 0,  c7_0); _mm512_storeu_ps(C + 7*ldc_elems + 16, c7_1);
}

static void avx512bf16_gemm_aligned(int M, int N, int K,
                    const bf16_t *A, int lda,
                    const bf16_t *B, int ldb,
                    float *C, int ldc)
{
    int mc = AVX_MC, nc = AVX_NC, kc = AVX_KC;
    if (mc > M) mc = M; if (nc > N) nc = N; if (kc > K) kc = K;
    mc -= mc % AVX_MR; if (mc == 0) mc = AVX_MR;
    nc -= nc % AVX_NR; if (nc == 0) nc = AVX_NR;
    kc -= kc % 2;      if (kc == 0) kc = 2;

    ensure_buffers(mc, nc, kc);

    for (int i = 0; i < M; i++) memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));

    for (int jc = 0; jc < N; jc += nc) {
        int cnc = (jc + nc <= N) ? nc : (N - jc);
        for (int pc = 0; pc < K; pc += kc) {
            int ckc = (pc + kc <= K) ? kc : (K - pc);
            pack_B(B + (size_t)pc * ldb + jc, ldb, ckc, cnc, g_Bpack);

            for (int ic = 0; ic < M; ic += mc) {
                int cmc = (ic + mc <= M) ? mc : (M - ic);
                pack_A(A + (size_t)ic * lda + pc, lda, cmc, ckc, g_Apack);

                for (int jr = 0; jr < cnc; jr += AVX_NR) {
                    bf16_t *Bblk = g_Bpack + jr * 2;
                    size_t ldb_bytes = (size_t)cnc * 4;
                    for (int ir = 0; ir < cmc; ir += AVX_MR) {
                        bf16_t *Ablk = g_Apack + (size_t)(ir / AVX_MR) * AVX_MR * ckc;
                        float *Cblk = C + (size_t)(ic + ir) * ldc + (jc + jr);
                        int zinit = (pc == 0) ? 1 : 0;
                        micro_8x32(Ablk, Bblk, ldb_bytes, Cblk, (size_t)ldc, ckc, zinit);
                    }
                }
            }
        }
    }
}

void avx512bf16_cleanup(void) {
    if (g_Apack) { free(g_Apack); g_Apack = NULL; g_Apack_cap = 0; }
    if (g_Bpack) { free(g_Bpack); g_Bpack = NULL; g_Bpack_cap = 0; }
}

#define ALIGN_AVX 32
static int round_up_avx(int x, int a) { return ((x + a - 1) / a) * a; }

void avx512bf16_gemm(int M, int N, int K,
                    const bf16_t *A, int lda,
                    const bf16_t *B, int ldb,
                    float *C, int ldc)
{
    /* M only needs to be a multiple of MR=8, N a multiple of NR=32, K a multiple of 2. */
    int Mp = round_up_avx(M, AVX_MR);
    int Np = round_up_avx(N, AVX_NR);
    int Kp = round_up_avx(K, 2);
    if (Mp == M && Np == N && Kp == K) {
        avx512bf16_gemm_aligned(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    bf16_t *Apad = aligned_alloc64((size_t)Mp * Kp * sizeof(bf16_t));
    bf16_t *Bpad = aligned_alloc64((size_t)Kp * Np * sizeof(bf16_t));
    float  *Cpad = aligned_alloc64((size_t)Mp * Np * sizeof(float));
    if (!Apad || !Bpad || !Cpad) { fprintf(stderr, "padding alloc failed\n"); abort(); }

    for (int i = 0; i < M; i++) {
        memcpy(Apad + (size_t)i * Kp, A + (size_t)i * lda, (size_t)K * sizeof(bf16_t));
        if (Kp > K) memset(Apad + (size_t)i * Kp + K, 0, (size_t)(Kp - K) * sizeof(bf16_t));
    }
    if (Mp > M) memset(Apad + (size_t)M * Kp, 0, (size_t)(Mp - M) * Kp * sizeof(bf16_t));
    for (int k = 0; k < K; k++) {
        memcpy(Bpad + (size_t)k * Np, B + (size_t)k * ldb, (size_t)N * sizeof(bf16_t));
        if (Np > N) memset(Bpad + (size_t)k * Np + N, 0, (size_t)(Np - N) * sizeof(bf16_t));
    }
    if (Kp > K) memset(Bpad + (size_t)K * Np, 0, (size_t)(Kp - K) * Np * sizeof(bf16_t));

    avx512bf16_gemm_aligned(Mp, Np, Kp, Apad, Kp, Bpad, Np, Cpad, Np);

    for (int i = 0; i < M; i++) {
        memcpy(C + (size_t)i * ldc, Cpad + (size_t)i * Np, (size_t)N * sizeof(float));
    }
    free(Apad); free(Bpad); free(Cpad);
}
