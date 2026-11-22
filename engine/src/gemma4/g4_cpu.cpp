// src/gemma4/g4_cpu.cpp - the Gemma 4 routed experts on the CPU: AVX2 + FMA + F16C kernels that give the bits of the
// scalar reference (g4_ref.hpp; the contract is in g4_fmt.hpp).  Compiled with /arch:AVX2 (-mavx2 -mfma -mf16c); the
// callers check cpu::available() first.
//
// Layout of the work: a weight row is unpacked once into byte vectors (K-quants: 8 sub-blocks of 32 per super-block;
// Q5_1 / Q8_0: 8 blocks of 32 per group) and every token's activation is multiplied against that unpacked row, so a
// verify window's tokens that share an expert read its weights once.  The integer sums of the 8 sub-blocks land in
// the 8 lanes of one vector (hsum8x8), which is exactly the lane structure of the contract's float accumulation.
#include "g4_cpu.hpp"

#include <cstdlib>
#include <cstring>

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#pragma fp_contract(off)
#else
#include <cpuid.h>
#endif

#include "g4_ref.hpp"

namespace strata::gemma4::cpu {

namespace {

void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int v[4];
    __cpuidex(v, leaf, sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) v[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
unsigned long long xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}
struct Caps {
    bool avx2 = false, vnni = false;
    Caps() {
        unsigned r[4];
        cpuid(0, 0, r);
        const unsigned max_leaf = r[0];
        cpuid(1, 0, r);
        const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1;
        const bool ymm = osxsave && (xgetbv0() & 6) == 6;
        if (max_leaf >= 7) {
            cpuid(7, 0, r);
            avx2 = ymm && avx && fma && f16c && ((r[1] >> 5) & 1);
            cpuid(7, 1, r);
            vnni = avx2 && ((r[0] >> 4) & 1);
        }
        if (const char* e = std::getenv("STRATA_G4_NO_VNNI"))
            if (e[0] == '1') vnni = false;
    }
};
const Caps& caps() {
    static const Caps c;
    return c;
}

void* aalloc(size_t n) {
#if defined(_MSC_VER)
    return _aligned_malloc((n + 63) / 64 * 64, 64);
#else
    return std::aligned_alloc(64, (n + 63) / 64 * 64);
#endif
}
void afree(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

/// fp16 bits -> float (F16C; MSVC has no scalar _cvtsh_ss).
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

/// The sums of 8 vectors of 8 int32, one per lane: [sum(p[0]), ..., sum(p[7])].
inline __m256i hsum8x8(const __m256i* p) {
    const __m256i s01 = _mm256_hadd_epi32(p[0], p[1]), s23 = _mm256_hadd_epi32(p[2], p[3]);
    const __m256i s45 = _mm256_hadd_epi32(p[4], p[5]), s67 = _mm256_hadd_epi32(p[6], p[7]);
    const __m256i t0 = _mm256_hadd_epi32(s01, s23), t1 = _mm256_hadd_epi32(s45, s67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20), _mm256_permute2x128_si256(t0, t1, 0x31));
}
/// ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
inline float reduce8(__m256 a) {
    const __m256 h = _mm256_hadd_ps(a, a);
    const __m128 lo = _mm256_castps256_ps128(h), hi = _mm256_extractf128_ps(h, 1);
    const __m128 l2 = _mm_add_ss(lo, _mm_movehdup_ps(lo)), h2 = _mm_add_ss(hi, _mm_movehdup_ps(hi));
    return _mm_cvtss_f32(_mm_add_ss(l2, h2));
}
/// 8 x int32: the sums of 4 adjacent products u8 * s8 (|product pair sums| stay inside int16 for every weight type here).
template <bool VNNI>
inline __m256i dot32(__m256i wu, __m256i xs) {
#if defined(__AVXVNNI__) || defined(_MSC_VER)
    if (VNNI) return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wu, xs);
#endif
    return _mm256_madd_epi16(_mm256_maddubs_epi16(wu, xs), _mm256_set1_epi16(1));
}

/// The 8 scales and 8 mins of a K super-block as int32 lanes (ggml's get_scale_min_k4 for j = 0..7).
inline void k_scales(const uint8_t* scales, __m256i& sc, __m256i& mn) {
    // in registers all the way: a vector load of bytes that scalar stores just wrote cannot be forwarded and stalls
    constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    uint32_t u0, u1, u2;
    std::memcpy(&u0, scales, 4);
    std::memcpy(&u1, scales + 4, 4);
    std::memcpy(&u2, scales + 8, 4);
    const uint64_t s8 = ((uint64_t) ((u2 & kmask2) | (((u0 >> 6) & kmask3) << 4)) << 32) | (u0 & kmask1);
    const uint64_t m8 = ((uint64_t) (((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4)) << 32) | (u1 & kmask1);
    sc = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) s8));
    mn = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) m8));
}

template <bool Q5, bool VNNI>
void gate_up_rows_k(const ExpertFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* out, int r0,
                    int r1) {
    const int nb = f.n_embd / 256;
    const size_t blk = Q5 ? sizeof(BlockQ5K) : sizeof(BlockQ4K);
    const __m256i m4 = _mm256_set1_epi8(0x0F), one8 = _mm256_set1_epi8(1);
    ref::Act a[kMaxTok];
    for (int t = 0; t < nt; ++t) a[t] = ref::act_view(const_cast<void*>(act[t]), f.n_embd);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = blob + (size_t) r * f.gu_row;
        __m256 acc[kMaxTok];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int B = 0; B < nb; ++B) {
            const uint8_t* b = row + (size_t) B * blk;
            uint16_t hd, hm;
            std::memcpy(&hd, b, 2);
            std::memcpy(&hm, b + 2, 2);
            const __m256 dv = _mm256_set1_ps(h2f(hd)), dminv = _mm256_set1_ps(h2f(hm));
            __m256i sc, mn;
            k_scales(b + 4, sc, mn);
            const uint8_t* qs = b + 16 + (Q5 ? 32 : 0);
            __m256i w[8];
            if (Q5) {
                const __m256i qh = _mm256_loadu_si256((const __m256i*) (b + 16));
                for (int p = 0; p < 4; ++p) {
                    const __m256i bits = _mm256_loadu_si256((const __m256i*) (qs + 32 * p));
                    const __m256i h0 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * p), one8);
                    const __m256i h1 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * p + 1), one8);
                    w[2 * p] = _mm256_or_si256(_mm256_and_si256(bits, m4), _mm256_slli_epi16(h0, 4));
                    w[2 * p + 1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), _mm256_slli_epi16(h1, 4));
                }
            } else {
                for (int p = 0; p < 4; ++p) {
                    const __m256i bits = _mm256_loadu_si256((const __m256i*) (qs + 32 * p));
                    w[2 * p] = _mm256_and_si256(bits, m4);
                    w[2 * p + 1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
                }
            }
            for (int t = 0; t < nt; ++t) {
                const int8_t* qx = a[t].q + (size_t) B * 256;
                __m256i p[8];
                for (int j = 0; j < 8; ++j) p[j] = dot32<VNNI>(w[j], _mm256_loadu_si256((const __m256i*) (qx + 32 * j)));
                const __m256i I = hsum8x8(p);
                const __m256 t1 = _mm256_cvtepi32_ps(_mm256_mullo_epi32(sc, I));
                const __m256 t2 = _mm256_cvtepi32_ps(_mm256_mullo_epi32(mn, _mm256_loadu_si256((const __m256i*) (a[t].b + B * 8))));
                const __m256 term = _mm256_fnmadd_ps(dminv, t2, _mm256_mul_ps(dv, t1));
                acc[t] = _mm256_fmadd_ps(_mm256_loadu_ps(a[t].s + B * 8), term, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = reduce8(acc[t]);
    }
}

/// 32 bytes, 0xFF where the bit of `x32` is set (ggml's bytes_from_bits_32).
inline __m256i bytes_from_bits_32(uint32_t x32) {
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

template <bool Q8, bool VNNI>
void down_rows_32(const ExpertFmt& f, const uint8_t* blob, const void* const* act_h, int nt, float* const* out, int r0,
                  int r1) {
    const int nb = f.n_ff / 32;
    const size_t blk = Q8 ? sizeof(BlockQ80) : sizeof(BlockQ51);
    const __m256i m4 = _mm256_set1_epi8(0x0F), b16 = _mm256_set1_epi8(0x10);
    ref::Act a[kMaxTok];
    for (int t = 0; t < nt; ++t) a[t] = ref::act_view(const_cast<void*>(act_h[t]), f.n_ff);
    const uint8_t* base = blob + f.down_off;
    alignas(32) static const int32_t kLaneMask[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    const int ib = (int) blk;
    const __m256i goff = _mm256_setr_epi32(0, ib, 2 * ib, 3 * ib, 4 * ib, 5 * ib, 6 * ib, 7 * ib);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = base + (size_t) r * f.d_row;
        __m256 acc[kMaxTok];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int g = 0; g < nb; g += 8) {
            const int n = nb - g < 8 ? nb - g : 8;   // blocks in this group (the lanes past n add exact zeros)
            const __m256i lane = _mm256_loadu_si256((const __m256i*) (kLaneMask + 8 - n));
            // the 8 blocks' first 4 bytes (Q5_1: d | m << 16; Q8_0: d and two weights) in one masked gather, then two
            // fp16 -> float conversions; no bytes go through the stack
            const __m256i hdr = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), (const int*) (row + (size_t) g * blk), goff, lane, 1);
            const __m256i dm = _mm256_permute4x64_epi64(
                _mm256_packus_epi32(_mm256_and_si256(hdr, _mm256_set1_epi32(0xffff)), _mm256_srli_epi32(hdr, 16)), 0xD8);
            __m256i w[8], sg[8];
            for (int j = 0; j < 8; ++j) {
                if (j >= n) {
                    w[j] = _mm256_setzero_si256();
                    sg[j] = _mm256_setzero_si256();
                    continue;
                }
                const uint8_t* b = row + (size_t) (g + j) * blk;
                if (Q8) {
                    const __m256i q = _mm256_loadu_si256((const __m256i*) (b + 2));
                    w[j] = _mm256_abs_epi8(q);   // |w| as u8; the sign goes to the activation (maddubs wants u8 * s8)
                    sg[j] = q;
                } else {
                    uint32_t qh;
                    std::memcpy(&qh, b + 4, 4);
                    const __m128i nib = _mm_loadu_si128((const __m128i*) (b + 8));
                    const __m256i lohi = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(nib, 4), nib), m4);
                    w[j] = _mm256_or_si256(lohi, _mm256_and_si256(bytes_from_bits_32(qh), b16));
                }
            }
            const __m256 dv = _mm256_cvtph_ps(_mm256_castsi256_si128(dm));
            const __m256 mv = _mm256_cvtph_ps(_mm256_extracti128_si256(dm, 1));   // (Q8_0: unused)
            for (int t = 0; t < nt; ++t) {
                const int8_t* qx = a[t].q + (size_t) g * 32;
                __m256i p[8];
                for (int j = 0; j < 8; ++j) {
                    if (j >= n) {
                        p[j] = _mm256_setzero_si256();
                        continue;
                    }
                    const __m256i x = _mm256_loadu_si256((const __m256i*) (qx + 32 * j));
                    p[j] = Q8 ? dot32<VNNI>(w[j], _mm256_sign_epi8(x, sg[j])) : dot32<VNNI>(w[j], x);
                }
                const __m256 fi = _mm256_cvtepi32_ps(hsum8x8(p));
                const __m256 pr = _mm256_mul_ps(dv, fi);
                __m256 term = pr;
                if (!Q8) {
                    const __m256i bs = _mm256_and_si256(_mm256_loadu_si256((const __m256i*) (a[t].b + g)), lane);
                    term = _mm256_fmadd_ps(mv, _mm256_cvtepi32_ps(bs), pr);
                }
                const __m256 sv = _mm256_and_ps(_mm256_loadu_ps(a[t].s + g), _mm256_castsi256_ps(lane));
                acc[t] = _mm256_fmadd_ps(sv, term, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) out[t][r] = reduce8(acc[t]);
    }
}

}  // namespace

bool available() noexcept { return caps().avx2; }
bool have_avx_vnni() noexcept { return caps().vnni; }

void quantize_act(const float* x, int64_t n, void* act) {
    const ref::Act a = ref::act_view(act, n);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* p = x + b * 32;
        const __m256 v0 = _mm256_loadu_ps(p), v1 = _mm256_loadu_ps(p + 8), v2 = _mm256_loadu_ps(p + 16),
                     v3 = _mm256_loadu_ps(p + 24);
        const __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                       _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        const __m256 mul = _mm256_set1_ps(inv);
        const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, mul)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, mul)),
                      i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, mul)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, mul));
        const __m256i p8 = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        _mm256_storeu_si256((__m256i*) (a.q + b * 32), _mm256_permutevar8x32_epi32(p8, perm));
        const __m256i s = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        a.s[b] = amax / 127.0f;
        a.b[b] = _mm_cvtsi128_si32(s4);
    }
}

void gate_up_rows(const ExpertFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const bool q5 = f.gu_type == kTypeQ5_K;
    if (caps().vnni) q5 ? gate_up_rows_k<true, true>(f, blob, act, nt, out, r0, r1) : gate_up_rows_k<false, true>(f, blob, act, nt, out, r0, r1);
    else q5 ? gate_up_rows_k<true, false>(f, blob, act, nt, out, r0, r1) : gate_up_rows_k<false, false>(f, blob, act, nt, out, r0, r1);
}

void down_rows(const ExpertFmt& f, const uint8_t* blob, const void* const* act_h, int nt, float* const* out, int r0, int r1) {
    const bool q8 = f.d_type == kTypeQ8_0;
    if (caps().vnni) q8 ? down_rows_32<true, true>(f, blob, act_h, nt, out, r0, r1) : down_rows_32<false, true>(f, blob, act_h, nt, out, r0, r1);
    else q8 ? down_rows_32<true, false>(f, blob, act_h, nt, out, r0, r1) : down_rows_32<false, false>(f, blob, act_h, nt, out, r0, r1);
}

void geglu_quantize(const float* gate_up, int n_ff, float* h, void* act_h) {
    const float* tab = ref::gelu_table();
    int j = 0;
    for (; j + 8 <= n_ff; j += 8) {
        alignas(16) uint16_t idx[8];
        _mm_store_si128((__m128i*) idx, _mm256_cvtps_ph(_mm256_loadu_ps(gate_up + j), _MM_FROUND_TO_NEAREST_INT));
        const __m256 g = _mm256_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]], tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]);
        _mm256_storeu_ps(h + j, _mm256_mul_ps(g, _mm256_loadu_ps(gate_up + n_ff + j)));
    }
    for (; j < n_ff; ++j) h[j] = tab[ref::fp32_to_fp16(gate_up[j])] * gate_up[n_ff + j];
    quantize_act(h, n_ff, act_h);
}

Scratch::~Scratch() {
    afree(gu);
    afree(h);
    afree(act_h);
}
void Scratch::reserve(int ff) {
    if (ff <= n_ff) return;
    afree(gu);
    afree(h);
    afree(act_h);
    n_ff = ff;
    gu = (float*) aalloc((size_t) kMaxTok * 2 * ff * sizeof(float));
    h = (float*) aalloc((size_t) ff * sizeof(float));
    act_h = (uint8_t*) aalloc((size_t) kMaxTok * act_bytes(ff));
}

void expert(const ExpertFmt& f, const uint8_t* blob, const void* const* act_x, int nt, float* const* out, Scratch& s) {
    s.reserve(f.n_ff);
    float* gu[kMaxTok];
    const void* ah[kMaxTok];
    for (int t = 0; t < nt; ++t) {
        gu[t] = s.gu + (size_t) t * 2 * f.n_ff;
        ah[t] = s.act_h + (size_t) t * act_bytes(f.n_ff);
    }
    gate_up_rows(f, blob, act_x, nt, gu, 0, 2 * f.n_ff);
    for (int t = 0; t < nt; ++t) geglu_quantize(gu[t], f.n_ff, s.h, const_cast<void*>(ah[t]));
    down_rows(f, blob, ah, nt, out, 0, f.n_embd);
}

}  // namespace strata::gemma4::cpu
