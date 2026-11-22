// src/gemma4/g4_ref.hpp - the scalar reference of the G4-A8 contract (g4_fmt.hpp).  Every other implementation (the
// AVX2 kernels in g4_cpu.cpp, the CUDA kernels in g4_kernels.cuh) must give these bits; `strata-g4 --selftest` checks
// it on real expert rows.  Not a hot path.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "g4_fmt.hpp"

namespace strata::gemma4::ref {

#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif

inline float f32_from_bits(uint32_t w) {
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}
inline uint32_t f32_to_bits(float f) {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    return w;
}
/// IEEE half -> float, exact.
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else {
            uint32_t m = man, shift = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1f) u = sign | 0x7f800000u | (man << 13);
    else u = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    return f32_from_bits(u);
}
/// float -> IEEE half bits, round to nearest even, overflow to infinity (what F16C and __float2half_rn give).
inline uint16_t fp32_to_fp16(float f) {
    const uint32_t x = f32_to_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));   // nan / inf
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);                                   // rounds to inf
    if (ax < 0x33000001u) return (uint16_t) sign;                                                // rounds to zero
    int e = (int) (ax >> 23) - 127;
    uint32_t m = (ax & 0x7fffffu) | 0x800000u;                                                   // 24-bit significand
    int shift;                                                                                   // bits dropped
    uint32_t base;
    if (e < -14) {                                                                               // subnormal half
        shift = 13 + (-14 - e);
        base = 0;
    } else {
        shift = 13;
        base = (uint32_t) (e + 15) << 10;
        m &= 0x7fffffu;
    }
    uint32_t r = shift >= 32 ? 0 : m >> shift;
    const uint32_t rem = shift >= 32 ? m : m & ((1u << shift) - 1u);
    const uint32_t half = shift >= 32 ? 0x80000000u : 1u << (shift - 1);
    if (shift < 32 && (rem > half || (rem == half && (r & 1u)))) ++r;
    return (uint16_t) (sign | (base + r));                                                       // a carry runs into the exponent
}

/// The GELU table: tab[bits] = gelu(fp16 value of bits), the tanh form, computed in double.
inline const float* gelu_table() {
    static const std::vector<float> tab = [] {
        std::vector<float> t(65536);
        const double c = std::sqrt(2.0 / 3.14159265358979323846);
        for (uint32_t i = 0; i < 65536; ++i) {
            const double x = (double) fp16_to_fp32((uint16_t) i);
            float v;
            if (std::isnan(x)) v = (float) x;
            else if (std::isinf(x)) v = x > 0 ? 65504.0f : -0.0f;
            else v = (float) (0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x))));
            t[i] = v;
        }
        return t;
    }();
    return tab.data();
}

struct Act {
    int8_t* q;
    float* s;
    int32_t* b;
};
inline Act act_view(void* buf, int64_t n) {
    uint8_t* p = (uint8_t*) buf;
    return {(int8_t*) (p + act_q_off(n)), (float*) (p + act_s_off(n)), (int32_t*) (p + act_b_off(n))};
}

/// x (n floats, n % 32 == 0) -> its A8 activation.
inline void quantize_act(const float* x, int64_t n, void* act) {
    const Act a = act_view(act, n);
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = std::fabs(x[b * 32 + i]);
            if (v > amax) amax = v;
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int32_t q = (int32_t) std::nearbyint(x[b * 32 + i] * inv);
            a.q[b * 32 + i] = (int8_t) q;
            sum += q;
        }
        a.s[b] = amax / 127.0f;
        a.b[b] = sum;
    }
}

inline float reduce8(const float* a) { return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])); }

inline float dot_q4k(const uint8_t* row, int64_t n, const Act& a) {
    const BlockQ4K* w = (const BlockQ4K*) row;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t B = 0; B < n / 256; ++B) {
        const float d = fp16_to_fp32(w[B].d), dmin = fp16_to_fp32(w[B].dmin);
        for (int j = 0; j < 8; ++j) {
            int sc, m;
            k4_scale_min(j, w[B].scales, sc, m);
            const uint8_t* qs = w[B].qs + (j / 2) * 32;
            const int8_t* qx = a.q + B * 256 + j * 32;
            int32_t I = 0;
            for (int i = 0; i < 32; ++i) I += (int32_t) ((j & 1) ? qs[i] >> 4 : qs[i] & 0xF) * (int32_t) qx[i];
            const float t1 = (float) (sc * I), t2 = (float) (m * a.b[B * 8 + j]);
            const float p = d * t1;
            const float term = std::fma(-dmin, t2, p);
            acc[j] = std::fma(a.s[B * 8 + j], term, acc[j]);
        }
    }
    return reduce8(acc);
}

inline float dot_q5k(const uint8_t* row, int64_t n, const Act& a) {
    const BlockQ5K* w = (const BlockQ5K*) row;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t B = 0; B < n / 256; ++B) {
        const float d = fp16_to_fp32(w[B].d), dmin = fp16_to_fp32(w[B].dmin);
        for (int j = 0; j < 8; ++j) {
            int sc, m;
            k4_scale_min(j, w[B].scales, sc, m);
            const uint8_t* qs = w[B].qs + (j / 2) * 32;
            const int8_t* qx = a.q + B * 256 + j * 32;
            int32_t I = 0;
            for (int i = 0; i < 32; ++i) {
                const int lo = (j & 1) ? qs[i] >> 4 : qs[i] & 0xF;
                const int hi = (w[B].qh[i] >> j) & 1;
                I += (int32_t) (lo + 16 * hi) * (int32_t) qx[i];
            }
            const float t1 = (float) (sc * I), t2 = (float) (m * a.b[B * 8 + j]);
            const float p = d * t1;
            const float term = std::fma(-dmin, t2, p);
            acc[j] = std::fma(a.s[B * 8 + j], term, acc[j]);
        }
    }
    return reduce8(acc);
}

inline float dot_q51(const uint8_t* row, int64_t n, const Act& a) {
    const BlockQ51* w = (const BlockQ51*) row;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t b = 0; b < n / 32; ++b) {
        const float d = fp16_to_fp32(w[b].d), m = fp16_to_fp32(w[b].m);
        uint32_t qh;
        std::memcpy(&qh, w[b].qh, 4);
        const int8_t* qx = a.q + b * 32;
        int32_t I = 0;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (w[b].qs[j] & 0x0F) | (int) (((qh >> j) & 1u) << 4);
            const int x1 = (w[b].qs[j] >> 4) | (int) (((qh >> (j + 16)) & 1u) << 4);
            I += x0 * (int32_t) qx[j] + x1 * (int32_t) qx[j + 16];
        }
        const float p = d * (float) I;
        const float term = std::fma(m, (float) a.b[b], p);
        acc[b & 7] = std::fma(a.s[b], term, acc[b & 7]);
    }
    return reduce8(acc);
}

inline float dot_q80(const uint8_t* row, int64_t n, const Act& a) {
    const BlockQ80* w = (const BlockQ80*) row;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t b = 0; b < n / 32; ++b) {
        const float d = fp16_to_fp32(w[b].d);
        const int8_t* qx = a.q + b * 32;
        int32_t I = 0;
        for (int i = 0; i < 32; ++i) I += (int32_t) w[b].qs[i] * (int32_t) qx[i];
        const float term = d * (float) I;
        acc[b & 7] = std::fma(a.s[b], term, acc[b & 7]);
    }
    return reduce8(acc);
}

inline float dot(int type, const uint8_t* row, int64_t n, const Act& a) {
    switch (type) {
        case kTypeQ4_K: return dot_q4k(row, n, a);
        case kTypeQ5_K: return dot_q5k(row, n, a);
        case kTypeQ5_1: return dot_q51(row, n, a);
        default: return dot_q80(row, n, a);
    }
}

/// h[j] = gelu_tab[fp16(gate_j)] * up_j from the 2 * n_ff gate/up outputs.
inline void geglu(const float* gate_up, int n_ff, float* h) {
    const float* tab = gelu_table();
    for (int j = 0; j < n_ff; ++j) h[j] = tab[fp32_to_fp16(gate_up[j])] * gate_up[n_ff + j];
}

/// One expert for one token: out[n_embd] = down(geglu(gate_up(act_x))).  `act_x` is the A8 activation of the layer's
/// expert input (n_embd values).  The per-expert scale and the router weight are applied by the combine, not here.
inline void expert(const ExpertFmt& f, const uint8_t* blob, const void* act_x, float* out) {
    std::vector<float> gu((size_t) 2 * f.n_ff), h((size_t) f.n_ff);
    const Act ax = act_view(const_cast<void*>(act_x), f.n_embd);
    for (int r = 0; r < 2 * f.n_ff; ++r) gu[(size_t) r] = dot(f.gu_type, blob + (size_t) r * f.gu_row, f.n_embd, ax);
    geglu(gu.data(), f.n_ff, h.data());
    std::vector<uint8_t> ah(act_bytes(f.n_ff));
    quantize_act(h.data(), f.n_ff, ah.data());
    const Act a = act_view(ah.data(), f.n_ff);
    for (int r = 0; r < f.n_embd; ++r) out[r] = dot(f.d_type, blob + f.down_off + (size_t) r * f.d_row, f.n_ff, a);
}

}  // namespace strata::gemma4::ref
