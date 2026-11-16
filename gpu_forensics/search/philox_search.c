/* Brute-force Philox seed search over sign patterns (CPU, OpenMP).
 *
 * For every seed in [seed_lo, seed_hi) and every layout group g, regenerate the
 * sign of the value PyTorch's CUDA RNG writes at each sampled position and count
 * agreements with the observed signs.  Signs need only the Philox output word:
 *   normal_  (curand_normal4, Box-Muller): even lane -> sin(v) > 0 <=> y < 2^31,
 *            odd lane -> cos(v) > 0 <=> y < 2^30 || y >= 3*2^30, with y = word 1
 *            (lanes 0,1) or word 3 (lanes 2,3)
 *   uniform_ / trunc_normal_ : value above centre <=> word[lane] >= 2^31
 * A seed is rejected as soon as it exceeds the allowed number of mismatches.
 */
#include <stdint.h>

#define M0 0xD2511F53u
#define M1 0xCD9E8D57u
#define W0 0x9E3779B9u
#define W1 0xBB67AE85u

static inline void philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t k0, uint32_t k1, uint32_t out[4]) {
    for (int i = 0; i < 10; i++) {
        uint64_t p0 = (uint64_t)M0 * c0, p1 = (uint64_t)M1 * c2;
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0, n1 = (uint32_t)p1;
        uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1, n3 = (uint32_t)p0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

static inline int sign_bit(const uint32_t w[4], int lane, int dist) {
    if (dist == 0) {
        uint32_t y = lane < 2 ? w[1] : w[3];
        return (lane & 1) ? (y < 0x40000000u || y >= 0xC0000000u) : (y < 0x80000000u);
    }
    return w[lane] >= 0x80000000u;
}

/* t, call, lane, expect: [groups][n] row-major.  Returns the number of hits
 * written (capped at max_out); *total receives the uncapped count. */
int64_t philox_search(const uint64_t *t, const uint64_t *call, const int32_t *lane, const uint8_t *expect,
                      int32_t n, int32_t groups, int32_t dist, uint64_t seed_lo, uint64_t seed_hi,
                      int32_t max_miss, uint64_t *out_seed, int32_t *out_group, int32_t *out_score,
                      int64_t max_out, int64_t *total) {
    int64_t hits = 0;
#pragma omp parallel for schedule(dynamic, 65536)
    for (int64_t s = (int64_t)seed_lo; s < (int64_t)seed_hi; s++) {
        uint32_t k0 = (uint32_t)s, k1 = (uint32_t)((uint64_t)s >> 32), w[4];
        for (int32_t g = 0; g < groups; g++) {
            const uint64_t *tg = t + (int64_t)g * n, *cg = call + (int64_t)g * n;
            const int32_t *lg = lane + (int64_t)g * n;
            const uint8_t *eg = expect + (int64_t)g * n;
            int32_t miss = 0;
            for (int32_t j = 0; j < n && miss <= max_miss; j++) {
                philox((uint32_t)cg[j], (uint32_t)(cg[j] >> 32), (uint32_t)tg[j], (uint32_t)(tg[j] >> 32), k0, k1, w);
                miss += sign_bit(w, lg[j], dist) != eg[j];
            }
            if (miss <= max_miss) {
                int64_t slot;
#pragma omp atomic capture
                slot = hits++;
                if (slot < max_out) {
                    out_seed[slot] = (uint64_t)s;
                    out_group[slot] = g;
                    out_score[slot] = n - miss;
                }
            }
        }
    }
    *total = hits;
    return hits < max_out ? hits : max_out;
}
