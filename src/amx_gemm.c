/*
 * AMX-based bf16 x bf16 -> fp32 GEMM
 * Target: Intel Sapphire Rapids (Xeon Platinum 8490H)
 *
 * Micro-kernel: 32x32 fp32 output (2x2 of 16x16 AMX tiles)
 * Cache blocking: GotoBLAS-style with packed A (16-row chunks) and packed B (VNNI)
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "amx_util.h"

/* Tile indices */
#define TC00 0
#define TC01 1
#define TC10 2
#define TC11 3
#define TA0  4
#define TA1  5
#define TB0  6
#define TB1  7

/* Default cache-blocking parameters (must be multiples of 32 for K, 32 for M and N).
 * MR=32, NR=32. Can be overridden at runtime via env vars AMX_MC, AMX_NC, AMX_KC. */
#ifndef AMX_MC
#define AMX_MC 128
#endif
#ifndef AMX_NC
#define AMX_NC 1024
#endif
#ifndef AMX_KC
#define AMX_KC 512
#endif

#define MR 32
#define NR 32
#define KSTEP 32  /* K dimension per AMX inner step */

/* Runtime-tunable block sizes. */
static int rt_mc = AMX_MC, rt_nc = AMX_NC, rt_kc = AMX_KC;
static int rt_pfd = 4;
static int rt_init = 0;
static void rt_init_once(void) {
    if (rt_init) return;
    const char *e;
    if ((e = getenv("AMX_MC"))  != NULL) rt_mc = atoi(e);
    if ((e = getenv("AMX_NC"))  != NULL) rt_nc = atoi(e);
    if ((e = getenv("AMX_KC"))  != NULL) rt_kc = atoi(e);
    if ((e = getenv("AMX_PFD")) != NULL) rt_pfd = atoi(e);
    rt_init = 1;
}

/* Configure all eight tiles for the 32x32-bf16 -> 32x32-fp32 micro-kernel. */
static void amx_configure(void) {
    amx_tilecfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    cfg.start_row = 0;
    /* C tiles (fp32, 16 rows x 16 cols = 64 bytes) */
    for (int t = 0; t < 4; t++) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    /* A tiles (bf16, 16 rows x 32 cols = 64 bytes) */
    for (int t = 4; t < 6; t++) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    /* B tiles (bf16 VNNI, 16 K-pair rows x 64 bytes) */
    for (int t = 6; t < 8; t++) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    _tile_loadconfig(&cfg);
}

/* Pack A panel: copies A[ic:ic+mc, pc:pc+kc] (bf16, row-major, lda stride in bf16 elements)
 * into Apack with layout suitable for AMX: 16-row chunks of (16 rows x kc bf16) stored
 * contiguously (so a tileloadd over 16 rows with stride 2*kc bytes works).
 * mc must be a multiple of 16, kc must be a multiple of KSTEP=32.
 * Apack must have size mc * kc bf16. */
static void pack_A(const bf16_t *A, int lda, int mc, int kc, bf16_t *Apack) {
    /* Layout: Apack[ chunk_row(=mc/16) ][ row_in_chunk(=16) ][ k(=kc) ] */
    int n_chunks = mc / 16;
    for (int c = 0; c < n_chunks; c++) {
        const bf16_t *src = A + (size_t)(c * 16) * lda;
        bf16_t *dst = Apack + (size_t)c * 16 * kc;
        for (int r = 0; r < 16; r++) {
            memcpy(dst + (size_t)r * kc, src + (size_t)r * lda, (size_t)kc * sizeof(bf16_t));
        }
    }
}

/* Pack B panel: B[pc:pc+kc, jc:jc+nc] (bf16, row-major, ldb stride in bf16 elements)
 * -> Bpack in VNNI: layout is (kc/2) rows of (nc*2) bf16, where for each col group
 *    Bpack[k_pair][2*n + 0] = B[2*k_pair + 0][n]
 *    Bpack[k_pair][2*n + 1] = B[2*k_pair + 1][n]
 * Additionally, we tile by NR=32 columns so that each NR-block is contiguous (16 K-pair rows x 64 bytes contiguous within a panel? no — we keep stride as nc*4 for the whole panel; that gives nice 64-byte tile loads).
 *
 * Simpler: just emit the entire panel with stride nc*4 bytes per K-pair row.
 * kc must be even and a multiple of KSTEP=32, nc must be a multiple of NR=32.
 */
static void pack_B(const bf16_t *B, int ldb, int kc, int nc, bf16_t *Bpack) {
    int k_pairs = kc / 2;
    /* SIMD VNNI pack: process 16 cols (= 32 bytes) of bf16 at a time using
     * 256-bit unpack-low/high + permute2x128 to produce 32 bf16 pairs in
     * VNNI order. */
    for (int kp = 0; kp < k_pairs; kp++) {
        const bf16_t *row0 = B + (size_t)(2 * kp + 0) * ldb;
        const bf16_t *row1 = B + (size_t)(2 * kp + 1) * ldb;
        bf16_t *dst = Bpack + (size_t)kp * (nc * 2);
        int n = 0;
        for (; n + 16 <= nc; n += 16) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(row0 + n));
            __m256i b = _mm256_loadu_si256((const __m256i*)(row1 + n));
            __m256i lo = _mm256_unpacklo_epi16(a, b);
            __m256i hi = _mm256_unpackhi_epi16(a, b);
            __m256i o0 = _mm256_permute2x128_si256(lo, hi, 0x20);
            __m256i o1 = _mm256_permute2x128_si256(lo, hi, 0x31);
            _mm256_storeu_si256((__m256i*)(dst + 2 * n + 0),  o0);
            _mm256_storeu_si256((__m256i*)(dst + 2 * n + 16), o1);
        }
        /* tail (nc not multiple of 16) */
        for (; n < nc; n++) {
            dst[2 * n + 0] = row0[n];
            dst[2 * n + 1] = row1[n];
        }
    }
}

/* Micro-kernel for a 32x32 block. Accumulates 32x32 fp32 output for kc K iterations.
 *
 * Apack_block: pointer to packed A block for these 32 rows (mc-relative 0..31),
 *              layout: 2 chunks of (16 rows x kc bf16), so Apack_block + 0
 *              gives chunk 0 (rows 0..15) and Apack_block + 16*kc gives chunk 1 (rows 16..31).
 *              Stride within a chunk = kc bf16 = 2*kc bytes.
 *
 * Bpack_block: pointer to packed B block for these 32 cols (nc-relative 0..31).
 *              The Bpack panel stride (one K-pair row) is nc*4 bytes (full panel width).
 *              Tile B0 starts at Bpack_block + 0, tile B1 at Bpack_block + 64 (in bytes).
 *
 * C_block:    pointer to output C[ir, jr] in row-major, stride ldc fp32 = ldc*4 bytes.
 *
 * accumulate_C: if 0, zero the C tiles before; if 1, load existing C from memory.
 * store_C:     if 1, store accumulated C tiles back to memory at the end.
 *
 * For inner-k accumulation we typically keep C in tiles across multiple kc steps. Caller
 * decides via accumulate_C / store_C flags. For simplicity here we let caller handle
 * loading and storing in the K-outer loop driver (this routine works for a single kc panel
 * with optional load/store).
 */
static inline void amx_micro_kernel_32x32(
    const bf16_t *Apack_block,
    const bf16_t *Bpack_block, size_t ldb_bytes,
    float *C_block, size_t ldc_bytes,
    int kc,
    int accumulate_C, int store_C)
{
    if (accumulate_C) {
        _tile_loadd(TC00, C_block, ldc_bytes);
        _tile_loadd(TC01, (char*)C_block + 64, ldc_bytes);
        _tile_loadd(TC10, (char*)C_block + 16 * ldc_bytes, ldc_bytes);
        _tile_loadd(TC11, (char*)C_block + 16 * ldc_bytes + 64, ldc_bytes);
    } else {
        _tile_zero(TC00);
        _tile_zero(TC01);
        _tile_zero(TC10);
        _tile_zero(TC11);
    }

    const int steps = kc / KSTEP;
    /* Apack stride within chunk: kc bf16 = 2*kc bytes */
    const size_t la = (size_t)kc * 2;
    /* Apack: chunk 0 starts at offset 0, chunk 1 starts at offset 16*kc bf16 = 32*kc bytes */
    const bf16_t *Achunk0 = Apack_block;
    const bf16_t *Achunk1 = Apack_block + (size_t)16 * kc;

    const int PFD = rt_pfd;

    for (int s = 0; s < steps; s++) {
        /* A tile bytes offset within chunk: s * KSTEP bf16 = s * 64 bytes */
        const char *Ap0 = (const char*)Achunk0 + (size_t)s * KSTEP * sizeof(bf16_t);
        const char *Ap1 = (const char*)Achunk1 + (size_t)s * KSTEP * sizeof(bf16_t);
        /* B pointer: row offset s * 16 K-pair rows times ldb_bytes */
        const char *Bp = (const char*)Bpack_block + (size_t)s * 16 * ldb_bytes;

        /* Light SW prefetch: a few lines per tile, PFD steps ahead.
         * Each tile spans 16 rows at stride ldb_bytes (B) or la (A).
         * We prefetch only every 4th row to keep prefetch instruction count low. */
        if (PFD > 0 && s + PFD < steps) {
            const char *Bp_pf = (const char*)Bpack_block + (size_t)(s + PFD) * 16 * ldb_bytes;
            _mm_prefetch(Bp_pf,                            _MM_HINT_T0);
            _mm_prefetch(Bp_pf +  4 * ldb_bytes,           _MM_HINT_T0);
            _mm_prefetch(Bp_pf +  8 * ldb_bytes,           _MM_HINT_T0);
            _mm_prefetch(Bp_pf + 12 * ldb_bytes,           _MM_HINT_T0);
            _mm_prefetch(Bp_pf + 64,                       _MM_HINT_T0);
            _mm_prefetch(Bp_pf +  4 * ldb_bytes + 64,      _MM_HINT_T0);
            _mm_prefetch(Bp_pf +  8 * ldb_bytes + 64,      _MM_HINT_T0);
            _mm_prefetch(Bp_pf + 12 * ldb_bytes + 64,      _MM_HINT_T0);
        }

        _tile_loadd(TA0, Ap0, la);
        _tile_loadd(TB0, Bp, ldb_bytes);
        _tile_dpbf16ps(TC00, TA0, TB0);
        _tile_loadd(TB1, Bp + 64, ldb_bytes);
        _tile_dpbf16ps(TC01, TA0, TB1);
        _tile_loadd(TA1, Ap1, la);
        _tile_dpbf16ps(TC10, TA1, TB0);
        _tile_dpbf16ps(TC11, TA1, TB1);
    }

    if (store_C) {
        _tile_stored(TC00, C_block, ldc_bytes);
        _tile_stored(TC01, (char*)C_block + 64, ldc_bytes);
        _tile_stored(TC10, (char*)C_block + 16 * ldc_bytes, ldc_bytes);
        _tile_stored(TC11, (char*)C_block + 16 * ldc_bytes + 64, ldc_bytes);
    }
}

/* Buffers for packed panels. Reused across calls. */
static bf16_t *g_Apack = NULL;       /* big A panel: M * kc bf16 (pre-packed across all ic for one pc) */
static bf16_t *g_Bpack = NULL;       /* one B panel: kc * nc bf16 */
static size_t g_Apack_capacity = 0;  /* in bf16 elements */
static size_t g_Bpack_capacity = 0;

static void ensure_buffers(size_t a_need, size_t b_need) {
    if (a_need > g_Apack_capacity) {
        if (g_Apack) free(g_Apack);
        g_Apack = (bf16_t*)aligned_alloc64(a_need * sizeof(bf16_t) + 128);
        g_Apack_capacity = a_need;
    }
    if (b_need > g_Bpack_capacity) {
        if (g_Bpack) free(g_Bpack);
        g_Bpack = (bf16_t*)aligned_alloc64(b_need * sizeof(bf16_t) + 128);
        g_Bpack_capacity = b_need;
    }
}

/* Internal: aligned dimensions only (M, N, K multiples of 32). */
static void amx_gemm_aligned(int M, int N, int K,
                             const bf16_t *A, int lda,
                             const bf16_t *B, int ldb,
                             float *C, int ldc)
{
    rt_init_once();
    int mc = rt_mc, nc = rt_nc, kc = rt_kc;
    if (mc > M) mc = M;
    if (nc > N) nc = N;
    if (kc > K) kc = K;
    /* Ensure mc is multiple of MR, nc of NR, kc of KSTEP */
    mc -= mc % MR; if (mc == 0) mc = MR;
    nc -= nc % NR; if (nc == 0) nc = NR;
    kc -= kc % KSTEP; if (kc == 0) kc = KSTEP;

    /* big A pack: pre-pack all M rows for current pc panel. */
    size_t big_Apack_need = (size_t)M * kc;
    size_t Bpack_need = (size_t)kc * nc;
    ensure_buffers(big_Apack_need, Bpack_need);

    amx_configure();

    /* Note: C is overwritten by the kernel at pc==0 (tile_zero + accumulate + store),
     * so a pre-zero is not needed. */

    /* Loop ordering: pc outer; pre-pack ALL A for this pc into g_Apack once;
     * then jc / ic / micro-kernel reuse g_Apack across all jc. */
    for (int pc = 0; pc < K; pc += kc) {
        int cur_kc = (pc + kc <= K) ? kc : (K - pc);

        /* Pre-pack all A panels for this pc into the big buffer. */
        for (int ic = 0; ic < M; ic += mc) {
            int cur_mc = (ic + mc <= M) ? mc : (M - ic);
            pack_A(A + (size_t)ic * lda + pc, lda, cur_mc, cur_kc,
                   g_Apack + (size_t)ic * cur_kc);
        }

        for (int jc = 0; jc < N; jc += nc) {
            int cur_nc = (jc + nc <= N) ? nc : (N - jc);
            pack_B(B + (size_t)pc * ldb + jc, ldb, cur_kc, cur_nc, g_Bpack);

            for (int ic = 0; ic < M; ic += mc) {
                int cur_mc = (ic + mc <= M) ? mc : (M - ic);
                bf16_t *Ap_ic = g_Apack + (size_t)ic * cur_kc;

                for (int jr = 0; jr < cur_nc; jr += NR) {
                    bf16_t *Bblk = g_Bpack + jr * 2;
                    size_t ldb_bytes = (size_t)cur_nc * 4;
                    for (int ir = 0; ir < cur_mc; ir += MR) {
                        bf16_t *Ablk = Ap_ic + (size_t)(ir / 16) * 16 * cur_kc;
                        float *Cblk = C + (size_t)(ic + ir) * ldc + (jc + jr);
                        size_t ldc_bytes = (size_t)ldc * 4;

                        int acc = (pc == 0) ? 0 : 1;
                        amx_micro_kernel_32x32(Ablk, Bblk, ldb_bytes,
                                               Cblk, ldc_bytes, cur_kc,
                                               acc, 1);
                    }
                }
            }
        }
    }

    _tile_release();
}

void amx_gemm_cleanup(void) {
    if (g_Apack) { free(g_Apack); g_Apack = NULL; g_Apack_capacity = 0; }
    if (g_Bpack) { free(g_Bpack); g_Bpack = NULL; g_Bpack_capacity = 0; }
}

/* Public entry: handles arbitrary sizes by padding to multiples of 32 internally.
 * Padding overhead is included in the timed call. */
#define ALIGN 32
static int round_up(int x, int a) { return ((x + a - 1) / a) * a; }

void amx_gemm_bf16(int M, int N, int K,
                   const bf16_t *A, int lda,
                   const bf16_t *B, int ldb,
                   float *C, int ldc)
{
    int Mp = round_up(M, ALIGN), Np = round_up(N, ALIGN), Kp = round_up(K, ALIGN);
    if (Mp == M && Np == N && Kp == K) {
        amx_gemm_aligned(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    /* Pad A: Mp x Kp, fill with zeros in padding rows/cols */
    bf16_t *Apad = aligned_alloc64((size_t)Mp * Kp * sizeof(bf16_t));
    bf16_t *Bpad = aligned_alloc64((size_t)Kp * Np * sizeof(bf16_t));
    float  *Cpad = aligned_alloc64((size_t)Mp * Np * sizeof(float));
    if (!Apad || !Bpad || !Cpad) { fprintf(stderr, "padding alloc failed\n"); abort(); }

    /* Copy A; zero padding cols within rows and zero padding rows entirely. */
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

    amx_gemm_aligned(Mp, Np, Kp, Apad, Kp, Bpad, Np, Cpad, Np);

    /* Copy back the original MxN block of C. */
    for (int i = 0; i < M; i++) {
        memcpy(C + (size_t)i * ldc, Cpad + (size_t)i * Np, (size_t)N * sizeof(float));
    }

    free(Apad); free(Bpad); free(Cpad);
}
