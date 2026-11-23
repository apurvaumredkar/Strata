// src/gemma4/g4_kernels.cuh - the Gemma 4 engine path's own kernels for everything but the routed experts
// (g4_expert.cuh) and the dense quantized mat-vecs (strata's native MMVQ): the token embedding, the RMS norms, the
// attention (per-head QK norm, NEOX RoPE, a ring cache for the sliding-window layers), the router and the layer's
// tail.  These run on the GPU only, so their arithmetic needs no CPU twin; it is fixed by this file.
//
// Rows: a forward runs T <= 8 rows, the consecutive positions of one conversation (a decode step, a verify window, a
// piece of the prompt).  Row t may read the cache cells of the rows before it in the same forward.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::gemma4::gpu {

constexpr int kThreads = 256;

__device__ __forceinline__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// x[t] = (d * q) * scale, the Q8_0 embedding row of tok[t].  grid (n / 32, T), block 32.
__global__ void k_embed(const uint8_t* embd, const int32_t* tok, float* x, int n, float scale) {
    const uint8_t* b = embd + (size_t) tok[blockIdx.y] * (size_t) (n / 32) * 34 + (size_t) blockIdx.x * 34;
    const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    x[(size_t) blockIdx.y * n + blockIdx.x * 32 + threadIdx.x] = (d * (float) (int8_t) b[2 + threadIdx.x]) * scale;
}

/// y[t] = rms(x[t]) * w.  grid T, block 256.
__global__ void k_rms_w(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] * r) * w[i];
}

/// The rotary table of a forward's rows: cs[t][p] = (cos, sin) of pair p at position pos[t], llama.cpp's sequence
/// (theta starts at the position and is multiplied by base^(-2 / n_rot) once per pair; the angle is theta / factor).
/// grid T, block n_rot / 2 (<= 256).
__global__ void k_rope_tab(const int32_t* pos, const float* factors, float2* cs, int n_rot, float theta_scale) {
    const int p = threadIdx.x;
    float th = (float) pos[blockIdx.x];
    for (int i = 0; i < p; ++i) th *= theta_scale;
    const float a = factors ? th / factors[p] : th;
    float sn, c;
    sincosf(a, &sn, &c);
    cs[(size_t) blockIdx.x * (n_rot / 2) + p] = make_float2(c, sn);
}

/// The attention layer's per-row preparation.  Block (head, t): heads [0, H) are the query heads (RMS over the head,
/// its weight, RoPE; kept as floats), [H, H + HKV) the key heads (the same, into the cache as fp16), [H + HKV,
/// H + 2 HKV) the value heads (RMS only, into the cache).  `v_in` is the value projection, or the key projection when
/// the layer has none.  Row t's cache cell is cell[t].  grid (H + 2 HKV, T), block 256, HD 256 or 512.
__global__ void k_attn_prep(const float* q_in, const float* k_in, const float* v_in, const float* qw, const float* kw,
                            const float2* cs, float* q_out, __half* kc, __half* vc, const int32_t* cell, int H, int HKV,
                            int HD, int cap, float eps) {
    __shared__ float sh[kThreads];
    __shared__ float buf[512];
    const int head = blockIdx.x, t = blockIdx.y;
    const int kind = head < H ? 0 : head < H + HKV ? 1 : 2;
    const int hh = kind == 0 ? head : kind == 1 ? head - H : head - H - HKV;
    const float* src = kind == 0 ? q_in + ((size_t) t * H + hh) * HD
                     : (kind == 1 ? k_in : v_in) + ((size_t) t * HKV + hh) * HD;
    float s = 0.f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) HD + eps);
    const float* w = kind == 0 ? qw : kw;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) buf[i] = kind == 2 ? src[i] * r : (src[i] * r) * w[i];
    __syncthreads();
    const int half = HD / 2;
    const float2* c = cs + (size_t) t * half;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float v = buf[i];
        if (kind != 2) {
            const int p = i < half ? i : i - half;
            const float2 e = c[p];
            v = i < half ? buf[i] * e.x - buf[i + half] * e.y : buf[i - half] * e.y + buf[i] * e.x;
        }
        if (kind == 0) q_out[((size_t) t * H + hh) * HD + i] = v;
        else if (kind == 1) kc[((size_t) hh * cap + cell[t]) * HD + i] = __float2half_rn(v);
        else vc[((size_t) hh * HD + i) * cap + cell[t]] = __float2half_rn(v);   // a dimension after the other
    }
}

/// Attention of one (query head, row) over the row's visible cells: n = count[t] cells, the j-th at
/// (first[t] + j) % cap (a ring for the sliding-window layers; first 0 and no wrap for the global ones).
/// out = softmax(q . k) v, scale 1 (the heads are normed).  `sc` is scratch: [T * H][cap].  grid (H, T), block 256.
__global__ void k_attn(const float* q, const __half* kc, const __half* vc, const int32_t* first, const int32_t* count,
                       float* sc, float* out, int H, int HKV, int HD, int cap) {
    __shared__ float sh[kThreads];
    __shared__ float qs[512];
    const int h = blockIdx.x, t = blockIdx.y, g = h / (H / HKV);
    const int n = count[t], f0 = first[t];
    const float* qr = q + ((size_t) t * H + h) * HD;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) qs[i] = qr[i];
    __syncthreads();
    float* s = sc + ((size_t) t * H + h) * cap;
    const __half* kb = kc + (size_t) g * cap * HD;
    const __half* vb = vc + (size_t) g * cap * HD;
    float mx = -3.0e38f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int c = f0 + j;
        if (c >= cap) c -= cap;
        const __half* kr = kb + (size_t) c * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a = fmaf(qs[i], __half2float(kr[i]), a);
        s[j] = a;
        mx = fmaxf(mx, a);
    }
    mx = block_max(mx, sh);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float w = expf(s[j] - mx);
        s[j] = w;
        z += w;
    }
    z = block_sum(z, sh);   // (its barriers also publish the weights)
    const float iz = 1.0f / z;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float a = 0.f;
        int c = f0;
        for (int j = 0; j < n; ++j) {
            a = fmaf(s[j], __half2float(vb[(size_t) i * cap + c]), a);
            if (++c == cap) c = 0;
        }
        out[((size_t) t * H + h) * HD + i] = a * iz;
    }
}

// ---- the attention the engine runs: k_attn's sums in k_attn's order (the same bits), with the memory read in wide
// pieces.  A GPU thread waits for every load, so k_attn's 2 bytes a step (a value of every cell, a stride apart) was
// what its time went to.  Here a key is loaded once for all the query heads and rows that share its KV head, 16 bytes
// at a time, and the values are stored a dimension after the other ([KV head][dimension][cell]), so a thread's sum
// over the cells reads 8 of them a load, and its weights 4 a load.
//   k_attn_score   every (query head, row) product of a cell's key; sc[(row, head)][cell]
//   k_attn_soft    k_attn's softmax of a (query head, row): the weights in `sc`, 1 / their sum in iz
//   k_attn_value   a (query head, row)'s sum over its cells in their order
// k_attn_score's block takes the kAttnPairs / (H / HKV) rows from TB * blockIdx.z; `nu` is the number of cells from the
// first row's first cell that the rows see between them (the rows are consecutive positions).  cap is a multiple of 8.
constexpr int kAttnPairs = 16;

__device__ __forceinline__ float half_bits(unsigned v) { return __half2float(__ushort_as_half((unsigned short) v)); }

/// The (row, query head of the KV head) pairs are numbered row * (H / HKV) + head; a block takes PB of them from
/// PB * blockIdx.z.  grid (HKV, ceil(nu / 256), ceil(T * (H / HKV) / PB)), block 256.
__global__ void k_attn_score(const float* q, const __half* kc, const int32_t* first, const int32_t* count, float* sc, int H,
                             int HKV, int HD, int cap, int T, int PB, int nu) {
    __shared__ float qs[kAttnPairs * 512];
    const int g = blockIdx.x, hpg = H / HKV, p0 = blockIdx.z * PB;
    const int np = min(PB, T * hpg - p0);
    for (int i = threadIdx.x; i < np * HD; i += blockDim.x) {
        const int pr = i / HD, d = i - pr * HD, pp = p0 + pr;
        qs[i] = q[((size_t) (pp / hpg) * H + g * hpg + pp % hpg) * HD + d];
    }
    __syncthreads();
    const int u = blockIdx.y * blockDim.x + threadIdx.x;
    if (u >= nu) return;
    int c = first[0] + u;
    if (c >= cap) c -= cap;
    bool in[kAttnPairs];
    bool any = false;
    for (int x = 0; x < np; ++x) {
        const int r = (p0 + x) / hpg;
        int j = c - first[r];
        if (j < 0) j += cap;
        in[x] = j < count[r];
        any = any || in[x];
    }
    if (!any) return;
    const uint4* kr = (const uint4*) (kc + ((size_t) g * cap + c) * HD);
    float a[kAttnPairs];
    for (int x = 0; x < kAttnPairs; ++x) a[x] = 0.f;
    for (int b = 0; b < HD / 8; ++b) {
        const uint4 w = kr[b];
        const unsigned ws[4] = {w.x, w.y, w.z, w.w};
        for (int e = 0; e < 8; ++e) {
            const float kv = half_bits(ws[e >> 1] >> (16 * (e & 1)));
            const int i = b * 8 + e;
            for (int x = 0; x < np; ++x) a[x] = fmaf(qs[x * HD + i], kv, a[x]);
        }
    }
    for (int x = 0; x < np; ++x) {
        const int pp = p0 + x;
        if (in[x]) sc[((size_t) (pp / hpg) * H + g * hpg + pp % hpg) * cap + c] = a[x];
    }
}

/// grid (H, T), block 256.
__global__ void k_attn_soft(const int32_t* first, const int32_t* count, float* sc, float* iz, int H, int cap) {
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, t = blockIdx.y, n = count[t], f0 = first[t];
    float* s = sc + ((size_t) t * H + h) * cap;
    float mx = -3.0e38f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int c = f0 + j;
        if (c >= cap) c -= cap;
        mx = fmaxf(mx, s[c]);
    }
    mx = block_max(mx, sh);
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        int c = f0 + j;
        if (c >= cap) c -= cap;
        const float w = expf(s[c] - mx);
        s[c] = w;
        z += w;
    }
    z = block_sum(z, sh);
    if (threadIdx.x == 0) iz[t * H + h] = 1.0f / z;
}

/// vt is [KV head][dimension][cell].  grid (H, T), block 256.
__global__ void k_attn_value(const __half* vt, const int32_t* first, const int32_t* count, const float* sc, const float* iz,
                             float* out, int H, int HKV, int HD, int cap) {
    const int h = blockIdx.x, t = blockIdx.y, g = h / (H / HKV);
    const int n = count[t], f0 = first[t];
    const float* s = sc + ((size_t) t * H + h) * cap;
    const float izv = iz[t * H + h];
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        const __half* v = vt + ((size_t) g * HD + i) * cap;
        float a = 0.f;
        int left = n, c = f0;
        while (left > 0) {
            const int end = min(c + left, cap);
            int cc = c;
            for (; cc < end && (cc & 7); ++cc) a = fmaf(s[cc], __half2float(v[cc]), a);
            for (; cc + 8 <= end; cc += 8) {
                const uint4 w = *(const uint4*) (v + cc);
                const float4 s0 = *(const float4*) (s + cc), s1 = *(const float4*) (s + cc + 4);
                a = fmaf(s0.x, half_bits(w.x), a);
                a = fmaf(s0.y, half_bits(w.x >> 16), a);
                a = fmaf(s0.z, half_bits(w.y), a);
                a = fmaf(s0.w, half_bits(w.y >> 16), a);
                a = fmaf(s1.x, half_bits(w.z), a);
                a = fmaf(s1.y, half_bits(w.z >> 16), a);
                a = fmaf(s1.z, half_bits(w.w), a);
                a = fmaf(s1.w, half_bits(w.w >> 16), a);
            }
            for (; cc < end; ++cc) a = fmaf(s[cc], __half2float(v[cc]), a);
            left -= end - c;
            c = 0;
        }
        out[((size_t) t * H + h) * HD + i] = a * izv;
    }
}

/// After the attention's output projection o: x1 = x + rms(o) * w_post, then the three norms of x1 the FFN block
/// reads: m = rms(x1) * w_ffn (the dense FFN), e = rms(x1) * w_pre2 (the routed experts), r = (rms(x1) * rs) *
/// w_rscale (the router, rs = 1 / sqrt(n)).  grid T, block 256.
__global__ void k_post_attn(const float* x, const float* o, const float* w_post, const float* w_ffn, const float* w_pre2,
                            const float* w_rscale, float* x1, float* m, float* e, float* r, int n, float eps, float rs) {
    __shared__ float sh[kThreads];
    const size_t off = (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += o[off + i] * o[off + i];
    s = block_sum(s, sh);
    const float ro = 1.0f / sqrtf(s / (float) n + eps);
    s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (o[off + i] * ro) * w_post[i] + x[off + i];
        x1[off + i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r1 = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x1[off + i] * r1;
        m[off + i] = v * w_ffn[i];
        e[off + i] = v * w_pre2[i];
        r[off + i] = (v * rs) * w_rscale[i];
    }
}

/// logits[t][e] = W[e] . r[t] for f32 W [n_expert][n], n a multiple of 4: a warp per expert, 8 experts a block.
/// grid (ceil(n_expert / 8), T), block 256.
__global__ void k_router_logits(const float* W, const float* r, float* logits, int n, int n_expert) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_expert) return;
    const float4* wr = (const float4*) (W + (size_t) o * n);
    const float4* xr = (const float4*) (r + (size_t) blockIdx.y * n);
    float acc = 0.f;
    for (int i = lane; i < n / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) logits[(size_t) blockIdx.y * n_expert + o] = acc;
}

/// The router's picks of a row: the K highest logits in falling order (a tie: the lower index first) and their
/// softmax weights renormalised over the picks.  grid T, block 256 (n_expert <= 256, K <= 16).
__global__ void k_router_topk(const float* logits, int32_t* ids, float* wts, int n_expert, int K) {
    __shared__ float lg[256];
    __shared__ float sel[16];
    const int t = blockIdx.x, e = threadIdx.x;
    if (e < n_expert) lg[e] = logits[(size_t) t * n_expert + e];
    __syncthreads();
    if (e < n_expert) {
        const float v = lg[e];
        int rank = 0;
        for (int j = 0; j < n_expert; ++j) rank += (lg[j] > v) || (lg[j] == v && j < e);
        if (rank < K) {
            ids[(size_t) t * K + rank] = e;
            sel[rank] = v;
        }
    }
    __syncthreads();
    if (e == 0) {
        float w[16], z = 0.f;
        for (int k = 0; k < K; ++k) {
            w[k] = expf(sel[k] - sel[0]);
            z += w[k];
        }
        for (int k = 0; k < K; ++k) wts[(size_t) t * K + k] = w[k] / z;
    }
}

/// h = gelu_tab[fp16(gate)] * up (the dense FFN's GeGLU; the table the experts use).
__global__ void k_geglu_dense(const float* gate, const float* up, const float* tab, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = tab[__half_as_ushort(__float2half_rn(gate[i]))] * up[i];
}

/// The layer's tail for one row: the K picks' outputs combined in pick order with their weights and per-expert
/// scales, the three norms and the residual:
///   moe = sum_k (w_k * scale[id_k]) * part_k        (the first product, then fma)
///   y   = rms(rms(mlp) * w_post1 + rms(moe) * w_post2) * w_post
///   x   = (y + x1) * out_scale
/// Pick k's part is row src[t * K + k] of parts_cpu when that is >= 0 (computed on the CPU pool), else row t * K + k of
/// parts_gpu.  grid T, block 256.
__global__ void k_ffn_out(const float* x1, const float* mlp, const float* parts_gpu, const float* parts_cpu,
                          const int32_t* src, const int32_t* ids, const float* wts, const float* scale, int K,
                          const float* w_post1, const float* w_post2, const float* w_post, float out_scale, float* x,
                          int n, float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    __shared__ float ck[16];
    const int t = blockIdx.x;
    if (threadIdx.x < K) {
        const int e = t * K + threadIdx.x;
        pk[threadIdx.x] = src[e] >= 0 ? parts_cpu + (size_t) src[e] * n : parts_gpu + (size_t) e * n;
        ck[threadIdx.x] = wts[e] * scale[ids[e]];
    }
    __syncthreads();
    const size_t off = (size_t) t * n;
    float sm = 0.f, se = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = ck[0] * pk[0][i];
        for (int k = 1; k < K; ++k) acc = fmaf(ck[k], pk[k][i], acc);
        x[off + i] = acc;   // moe, parked in the output row until the norms are known
        se += acc * acc;
        sm += mlp[off + i] * mlp[off + i];
    }
    sm = block_sum(sm, sh);
    se = block_sum(se, sh);
    const float rm = 1.0f / sqrtf(sm / (float) n + eps), re = 1.0f / sqrtf(se / (float) n + eps);
    float sc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float c = (mlp[off + i] * rm) * w_post1[i] + (x[off + i] * re) * w_post2[i];
        x[off + i] = c;
        sc += c * c;
    }
    sc = block_sum(sc, sh);
    const float rc = 1.0f / sqrtf(sc / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[off + i] = ((x[off + i] * rc) * w_post[i] + x1[off + i]) * out_scale;
}

/// The drafter's block, first half: x1 = x + rms(o) * w_post, m = rms(x1) * w_ffn.  grid T, block 256.
__global__ void k_res_norm2(const float* x, const float* o, const float* w_post, const float* w_ffn, float* x1, float* m, int n,
                            float eps) {
    __shared__ float sh[kThreads];
    const size_t off = (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += o[off + i] * o[off + i];
    s = block_sum(s, sh);
    const float ro = 1.0f / sqrtf(s / (float) n + eps);
    s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (o[off + i] * ro) * w_post[i] + x[off + i];
        x1[off + i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r1 = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) m[off + i] = (x1[off + i] * r1) * w_ffn[i];
}
/// The drafter's block, second half: x = (rms(f) * w + x1) * out_scale.  grid T, block 256.
__global__ void k_res_tail(const float* x1, const float* f, const float* w, float out_scale, float* x, int n, float eps) {
    __shared__ float sh[kThreads];
    const size_t off = (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += f[off + i] * f[off + i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[off + i] = ((f[off + i] * r) * w[i] + x1[off + i]) * out_scale;
}
/// dst[i] = src[i].
__global__ void k_copy(const float* src, float* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/// How sure a softmax over n floats is of its largest value: conf[0] = 1 / sum(exp(x - max)).  Two stages: grid 64
/// (kArgBlocks) block 256 writes each block's sum into ps; one block of 64 adds them.  Only the drafter's guesses are
/// weighed with it (how many to chain), never a token of the answer.
__global__ void k_conf1(const float* x, int n, const float* vmax, float* ps) {
    __shared__ float sv[kThreads];
    const float mx = vmax[0];
    float a = 0.f;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += 64 * kThreads) a += __expf(x[i] - mx);
    sv[threadIdx.x] = a;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ps[blockIdx.x] = sv[0];
}
__global__ void k_conf2(const float* ps, float* conf) {
    __shared__ float sv[64];
    sv[threadIdx.x] = ps[threadIdx.x];
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) conf[0] = 1.0f / sv[0];
}

/// Argmax of rows of n floats (the lower index wins a tie), two stages: stage 1 grid (kArgBlocks, T) block 256 writes
/// each block's best into pv / pi [T][kArgBlocks]; stage 2 grid T block kArgBlocks picks the row's.
constexpr int kArgBlocks = 64;
__global__ void k_argmax1(const float* x, int n, float* pv, int32_t* pi) {
    __shared__ float sv[kThreads];
    __shared__ int32_t si[kThreads];
    const float* xr = x + (size_t) blockIdx.y * n;
    const int stride = kArgBlocks * kThreads;
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
        const float v = xr[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float v = sv[threadIdx.x + s];
            const int i = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
                sv[threadIdx.x] = v;
                si[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pv[(size_t) blockIdx.y * kArgBlocks + blockIdx.x] = sv[0];
        pi[(size_t) blockIdx.y * kArgBlocks + blockIdx.x] = si[0];
    }
}
__global__ void k_argmax2(const float* pv, const int32_t* pi, int32_t* out, float* out_v) {
    __shared__ float sv[kArgBlocks];
    __shared__ int32_t si[kArgBlocks];
    sv[threadIdx.x] = pv[(size_t) blockIdx.x * kArgBlocks + threadIdx.x];
    si[threadIdx.x] = pi[(size_t) blockIdx.x * kArgBlocks + threadIdx.x];
    __syncthreads();
    for (int s = kArgBlocks / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float v = sv[threadIdx.x + s];
            const int i = si[threadIdx.x + s];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
                sv[threadIdx.x] = v;
                si[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = si[0];
        if (out_v) out_v[blockIdx.x] = sv[0];
    }
}

}  // namespace strata::gemma4::gpu
