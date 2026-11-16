// CUDA port of philox_search.c (same ABI).  Experimental: not yet validated on
// hardware; the CPU backend is the reference.  Build: nvcc -O3 -shared -Xcompiler -fPIC
#include <stdint.h>

#define M0 0xD2511F53u
#define M1 0xCD9E8D57u
#define W0 0x9E3779B9u
#define W1 0xBB67AE85u

__device__ __forceinline__ void philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                                       uint32_t k1, uint32_t out[4]) {
#pragma unroll
    for (int i = 0; i < 10; i++) {
        uint32_t hi0 = __umulhi(M0, c0), lo0 = M0 * c0;
        uint32_t hi1 = __umulhi(M1, c2), lo1 = M1 * c2;
        uint32_t n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

__device__ __forceinline__ int sign_bit(const uint32_t w[4], int lane, int dist) {
    if (dist == 0) {
        uint32_t y = lane < 2 ? w[1] : w[3];
        return (lane & 1) ? (y < 0x40000000u || y >= 0xC0000000u) : (y < 0x80000000u);
    }
    return w[lane] >= 0x80000000u;
}

__global__ void kernel(const uint64_t *t, const uint64_t *call, const int32_t *lane, const uint8_t *expect,
                       int32_t n, int32_t groups, int32_t dist, uint64_t seed_lo, uint64_t count,
                       int32_t max_miss, unsigned long long *hits, uint64_t *out_seed, int32_t *out_group,
                       int32_t *out_score, int64_t max_out) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t s = seed_lo + i;
        uint32_t k0 = (uint32_t)s, k1 = (uint32_t)(s >> 32), w[4];
        for (int32_t g = 0; g < groups; g++) {
            int32_t miss = 0;
            for (int32_t j = 0; j < n && miss <= max_miss; j++) {
                int64_t k = (int64_t)g * n + j;
                philox((uint32_t)call[k], (uint32_t)(call[k] >> 32), (uint32_t)t[k], (uint32_t)(t[k] >> 32), k0, k1, w);
                miss += sign_bit(w, lane[k], dist) != expect[k];
            }
            if (miss <= max_miss) {
                unsigned long long slot = atomicAdd(hits, 1ULL);
                if ((int64_t)slot < max_out) {
                    out_seed[slot] = s;
                    out_group[slot] = g;
                    out_score[slot] = n - miss;
                }
            }
        }
    }
}

extern "C" int64_t philox_search(const uint64_t *t, const uint64_t *call, const int32_t *lane,
                                 const uint8_t *expect, int32_t n, int32_t groups, int32_t dist,
                                 uint64_t seed_lo, uint64_t seed_hi, int32_t max_miss, uint64_t *out_seed,
                                 int32_t *out_group, int32_t *out_score, int64_t max_out, int64_t *total) {
    size_t m = (size_t)n * groups;
    uint64_t *dt, *dc, *ds;
    int32_t *dl, *dg, *dsc;
    uint8_t *de;
    unsigned long long *dh, h = 0;
    cudaMalloc(&dt, m * 8); cudaMalloc(&dc, m * 8); cudaMalloc(&dl, m * 4); cudaMalloc(&de, m);
    cudaMalloc(&ds, max_out * 8); cudaMalloc(&dg, max_out * 4); cudaMalloc(&dsc, max_out * 4);
    cudaMalloc(&dh, 8);
    cudaMemcpy(dt, t, m * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, call, m * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dl, lane, m * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(de, expect, m, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, &h, 8, cudaMemcpyHostToDevice);
    kernel<<<4096, 256>>>(dt, dc, dl, de, n, groups, dist, seed_lo, seed_hi - seed_lo, max_miss, dh, ds, dg, dsc,
                          max_out);
    cudaMemcpy(&h, dh, 8, cudaMemcpyDeviceToHost);
    int64_t written = (int64_t)h < max_out ? (int64_t)h : max_out;
    cudaMemcpy(out_seed, ds, written * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_group, dg, written * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_score, dsc, written * 4, cudaMemcpyDeviceToHost);
    cudaFree(dt); cudaFree(dc); cudaFree(dl); cudaFree(de); cudaFree(ds); cudaFree(dg); cudaFree(dsc); cudaFree(dh);
    *total = (int64_t)h;
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? written : -1;
}
