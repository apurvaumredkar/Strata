// src/gemma4/g4_expert.cuh - the Gemma 4 routed experts on the GPU: kernels that give the bits of the scalar
// reference (g4_ref.hpp; the contract is in g4_fmt.hpp).  The translation unit that includes this is compiled with
// -fmad=false: the only fused multiply-adds are the fmaf() calls written here, as on the CPU.
//
// The routed entries of a window are grouped per expert (the host builds the groups from the routing it reads back
// anyway): group g is an expert's blob in a VRAM slot and the `nt` entries (window row, pick) that use it.  A group's
// weight rows are read once for all of its entries.
//
//   k_quant_a8        float rows -> A8 activations (the layer's expert input: the CPU pool reads the same bytes)
//   k_gate_up<Q5>     gate/up rows of every group: gu[entry][2 * n_ff]
//   k_geglu_quant     h = gelu_tab[fp16(gate)] * up and its A8 activation, per entry
//   k_down<Q8>        down rows of every group: parts[entry][n_embd]
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "g4_fmt.hpp"

namespace strata::gemma4::gpu {

constexpr int kMaxTok = 8;       // entries of one group (rows of a window)

/// A group of entries on one expert.  `blob` is the expert's device slot ([gate_up rows | down rows], g4_fmt.hpp).
struct Group {
    const uint8_t* blob;
    int32_t nt;                  // entries in the group, 1..kMaxTok
    int32_t ent[kMaxTok];        // their entry indices (row * top_k + pick): the row of gu / act_h / parts they own
    int32_t row[kMaxTok];        // their window rows: the activation each reads
    int32_t pad_;
};

/// The device view of A8 activations of `n` values per row, `stride` bytes between rows (act_bytes(n) or more).
struct ActDev {
    uint8_t* base;
    size_t stride;
    int n;
    __device__ __forceinline__ const int8_t* q(int r) const { return (const int8_t*) (base + (size_t) r * stride); }
    __device__ __forceinline__ const float* s(int r) const { return (const float*) (base + (size_t) r * stride + act_s_off(n)); }
    __device__ __forceinline__ const int32_t* b(int r) const { return (const int32_t*) (base + (size_t) r * stride + act_b_off(n)); }
};

/// x[row][n] -> A8.  grid (n / 32, rows), block 32: one thread a block of 32 values would waste the warp, so a block of
/// 32 threads takes one 32-value block and each thread one value.
__global__ void k_quant_a8(const float* x, int ldx, ActDev a) {
    const int b = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
    const float v = x[(size_t) r * ldx + b * 32 + i];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q = __float2int_rn(v * inv);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    ((int8_t*) a.q(r))[b * 32 + i] = (int8_t) q;
    if (i == 0) {
        ((float*) a.s(r))[b] = amax / 127.0f;
        ((int32_t*) a.b(r))[b] = sum;
    }
}

__device__ __forceinline__ int dp4(unsigned w, int x, int c) { return __dp4a((int) w, x, c); }

/// Gate/up rows.  Four threads a row (thread p: sub-blocks 2p and 2p + 1 of every super-block), 32 rows a block of 128
/// threads.  grid (2 n_ff / 32, n_groups).
template <bool Q5>
__global__ void __launch_bounds__(128) k_gate_up(const Group* groups, ActDev ax, float* gu, int n_embd, int n_ff) {
    const Group& G = groups[blockIdx.y];
    const int r = blockIdx.x * 32 + (threadIdx.x >> 2), p = threadIdx.x & 3;   // 2 n_ff is a multiple of 32 (n_ff of 32): no partial block
    const int nt = G.nt, nb = n_embd / 256;
    const size_t blk = Q5 ? sizeof(BlockQ5K) : sizeof(BlockQ4K);
    const uint8_t* row = G.blob + (size_t) r * (size_t) nb * blk;
    float alo[kMaxTok], ahi[kMaxTok];
#pragma unroll
    for (int t = 0; t < kMaxTok; ++t) alo[t] = ahi[t] = 0.0f;
    for (int B = 0; B < nb; ++B) {
        const uint8_t* b = row + (size_t) B * blk;
        const uint32_t hdr = *(const uint32_t*) b;                 // d | dmin << 16 (fp16)
        const float d = __half2float(__ushort_as_half((unsigned short) (hdr & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (hdr >> 16)));
        int sc0, m0, sc1, m1;
        k4_scale_min(2 * p, b + 4, sc0, m0);
        k4_scale_min(2 * p + 1, b + 4, sc1, m1);
        const uint32_t* qs = (const uint32_t*) (b + 16 + (Q5 ? 32 : 0) + 32 * p);
        uint32_t lo[8], hi[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t w = qs[i];
            lo[i] = w & 0x0f0f0f0fu;
            hi[i] = (w >> 4) & 0x0f0f0f0fu;
        }
        if (Q5) {
            const uint32_t* qh = (const uint32_t*) (b + 16);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t h = qh[i];   // byte k: bit 2p is the fifth bit of sub-block 2p's value, bit 2p + 1 of 2p + 1's
                lo[i] |= ((h >> (2 * p)) & 0x01010101u) << 4;
                hi[i] |= ((h >> (2 * p + 1)) & 0x01010101u) << 4;
            }
        }
        const int g0 = B * 8 + 2 * p;   // the activation's block index of sub-block 2p
        for (int t = 0; t < nt; ++t) {
            const int ar = G.row[t];
            const int32_t* x0 = (const int32_t*) (ax.q(ar) + (size_t) g0 * 32);
            int I0 = 0, I1 = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                I0 = dp4(lo[i], x0[i], I0);
                I1 = dp4(hi[i], x0[8 + i], I1);
            }
            const float* as = ax.s(ar) + g0;
            const int32_t* ab = ax.b(ar) + g0;
            const float t0 = fmaf(-dmin, (float) (m0 * ab[0]), d * (float) (sc0 * I0));
            const float t1 = fmaf(-dmin, (float) (m1 * ab[1]), d * (float) (sc1 * I1));
            alo[t] = fmaf(as[0], t0, alo[t]);
            ahi[t] = fmaf(as[1], t1, ahi[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        float v = alo[t] + ahi[t];                       // a(2p) + a(2p + 1)
        v += __shfl_xor_sync(0xffffffffu, v, 1);         // (a0 + a1) + (a2 + a3) | (a4 + a5) + (a6 + a7)
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if (p == 0) gu[(size_t) G.ent[t] * (size_t) (2 * n_ff) + r] = v;
    }
}

/// h = gelu_tab[fp16(gate)] * up and its A8 activation, per entry.  grid (n_ff / 32, n_entries listed in `ents`),
/// block 32.
__global__ void k_geglu_quant(const int32_t* ents, const float* gu, const float* gelu_tab, ActDev ah, int n_ff) {
    const int e = ents[blockIdx.y], b = blockIdx.x, i = threadIdx.x;
    const float* g = gu + (size_t) e * (size_t) (2 * n_ff);
    const int j = b * 32 + i;
    const float v = gelu_tab[__half_as_ushort(__float2half_rn(g[j]))] * g[n_ff + j];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q = __float2int_rn(v * inv);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    ((int8_t*) ah.q(e))[j] = (int8_t) q;
    if (i == 0) {
        ((float*) ah.s(e))[b] = amax / 127.0f;
        ((int32_t*) ah.b(e))[b] = sum;
    }
}

/// Down rows.  Eight threads a row (thread l: blocks l, l + 8, l + 16, ...), 32 rows a block of 256 threads.
/// grid (n_embd / 32, n_groups).  The activation of an entry is row G.ent[t] of `ah`.
template <bool Q8>
__global__ void __launch_bounds__(256) k_down(const Group* groups, ActDev ah, float* parts, size_t down_off, int n_embd,
                                              int n_ff) {
    const Group& G = groups[blockIdx.y];
    const int r = blockIdx.x * 32 + (threadIdx.x >> 3), l = threadIdx.x & 7;   // n_embd is a multiple of 32: no partial block
    const int nt = G.nt, nb = n_ff / 32;
    const size_t blk = Q8 ? sizeof(BlockQ80) : sizeof(BlockQ51);
    const uint8_t* row = G.blob + down_off + (size_t) r * (size_t) nb * blk;
    float acc[kMaxTok];
#pragma unroll
    for (int t = 0; t < kMaxTok; ++t) acc[t] = 0.0f;
    for (int b = l; b < nb; b += 8) {
        const uint8_t* w = row + (size_t) b * blk;
        uint32_t q[8];
        float d, m = 0.0f;
        if (Q8) {
            // 34-byte blocks: only 2-byte aligned, read bytewise into words
            d = __half2float(__ushort_as_half((unsigned short) (w[0] | (w[1] << 8))));
#pragma unroll
            for (int i = 0; i < 8; ++i)
                q[i] = (uint32_t) w[2 + 4 * i] | ((uint32_t) w[3 + 4 * i] << 8) | ((uint32_t) w[4 + 4 * i] << 16) |
                       ((uint32_t) w[5 + 4 * i] << 24);
        } else {
            const uint32_t* u = (const uint32_t*) w;   // 24-byte blocks in 528-byte rows: 4-byte aligned
            d = __half2float(__ushort_as_half((unsigned short) (u[0] & 0xffffu)));
            m = __half2float(__ushort_as_half((unsigned short) (u[0] >> 16)));
            const uint32_t qh = u[1];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t ws = u[2 + i];
                // four bits -> bit 4 of four bytes: (x * 0x00204081) & 0x01010101 spreads bit k to byte k
                const uint32_t h0 = ((((qh >> (4 * i)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
                const uint32_t h1 = ((((qh >> (16 + 4 * i)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
                q[i] = (ws & 0x0f0f0f0fu) | h0;            // values 4i .. 4i + 3
                q[4 + i] = ((ws >> 4) & 0x0f0f0f0fu) | h1;   // values 16 + 4i .. 16 + 4i + 3
            }
        }
        for (int t = 0; t < nt; ++t) {
            const int e = G.ent[t];
            const int32_t* x = (const int32_t*) (ah.q(e) + (size_t) b * 32);
            int I = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) I = __dp4a((int) q[i], x[i], I);
            const float pr = d * (float) I;
            const float term = Q8 ? pr : fmaf(m, (float) ah.b(e)[b], pr);
            acc[t] = fmaf(ah.s(e)[b], term, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        float v = acc[t];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        if (l == 0) parts[(size_t) G.ent[t] * (size_t) n_embd + r] = v;
    }
}

}  // namespace strata::gemma4::gpu
