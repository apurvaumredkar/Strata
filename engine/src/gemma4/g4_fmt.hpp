// src/gemma4/g4_fmt.hpp - Gemma 4 26B A4B engine path: the weight block layouts it reads and the arithmetic contract
// of its routed-expert kernels.  Shared by the CPU kernels (g4_cpu.cpp), the CUDA kernels (g4_kernels.cuh) and the
// scalar reference (g4_ref.hpp).
//
// THE CONTRACT ("G4-A8").  An expert's output must not depend on where the expert was computed: an expert that has a
// VRAM slot runs on the GPU, one that has none runs on the CPU pool, and which is which changes while a request runs
// (the cache follows the routing).  So the three implementations give the SAME BITS for the same inputs:
//
//   activation   x (n floats, n a multiple of 32) -> blocks of 32: amax = max |x_i|, inv = 127 / amax (0 when amax is
//                0), q_i = rint(x_i * inv) as int8 (round half to even), s = amax / 127, bsum = sum q_i.
//   row dot      the integer sums are exact in any order.  The float steps are fixed (below, per weight type) and the
//                per-block terms are accumulated in 8 lanes (lane = block index mod 8, in rising block order, with
//                acc = fma(s, term, acc)), then reduced as ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)).
//   GeGLU        h = gelu_tab[fp16(gate)] * up, gelu_tab the 65,536-entry table of the tanh GELU at every fp16 value
//                (computed once on the host, the same table on both devices).
//
// Per weight type, for one block (or K sub-block) of 32 weights with integer sum I = sum q_w * q_x:
//   Q4_K / Q5_K  value = d * sc * q - dmin * m          term = fma(-dmin, float(m * bsum), d * float(sc * I))
//   Q5_1         value = d * q + m                      term = fma(m, float(bsum), d * float(I))
//   Q8_0         value = d * q                          term = d * float(I)
// Every int -> float conversion above is exact (the products stay below 2^24).  No fused multiply-add other than the
// ones written, on either device: the CUDA file is compiled with -fmad=false, the CPU file uses the intrinsics.
#pragma once

#include <cstddef>
#include <cstdint>

// host and device: the CUDA kernels call these too
#if defined(__CUDACC__)
#define G4_HD __host__ __device__
#else
#define G4_HD
#endif

namespace strata::gemma4 {

// ggml type ids (the GGUF file format)
constexpr int kTypeF32 = 0, kTypeQ5_1 = 7, kTypeQ8_0 = 8, kTypeQ4_K = 12, kTypeQ5_K = 13;

#pragma pack(push, 1)
struct BlockQ4K {      // 256 weights
    uint16_t d, dmin;  // fp16
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlockQ5K {      // 256 weights
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct BlockQ51 {      // 32 weights
    uint16_t d, m;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct BlockQ80 {      // 32 weights
    uint16_t d;
    int8_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K");
static_assert(sizeof(BlockQ5K) == 176, "block_q5_K");
static_assert(sizeof(BlockQ51) == 24, "block_q5_1");
static_assert(sizeof(BlockQ80) == 34, "block_q8_0");

/// Bytes of a row of `n` weights of ggml type `t` (0: a type the expert kernels do not take, or n not whole blocks).
G4_HD constexpr size_t row_bytes(int t, int64_t n) {
    return t == kTypeQ4_K ? (n % 256 ? 0 : (size_t) (n / 256) * sizeof(BlockQ4K))
         : t == kTypeQ5_K ? (n % 256 ? 0 : (size_t) (n / 256) * sizeof(BlockQ5K))
         : t == kTypeQ5_1 ? (n % 32 ? 0 : (size_t) (n / 32) * sizeof(BlockQ51))
         : t == kTypeQ8_0 ? (n % 32 ? 0 : (size_t) (n / 32) * sizeof(BlockQ80))
         : 0;
}
constexpr bool gate_up_type_ok(int t) { return t == kTypeQ4_K || t == kTypeQ5_K; }
constexpr bool down_type_ok(int t) { return t == kTypeQ5_1 || t == kTypeQ8_0; }

/// The 6-bit scale and min of sub-block j (0..7) of a K super-block (ggml's get_scale_min_k4).
G4_HD inline void k4_scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// One expert's geometry: gate and up fused as 2 * n_ff rows of n_embd weights (gate rows first), then n_embd down
/// rows of n_ff weights.  A blob is [gate_up rows | down rows], `stride` bytes apart (a multiple of 64).
struct ExpertFmt {
    int gu_type = 0, d_type = 0;
    int n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;   // bytes per row
    size_t gu_bytes = 0, d_bytes = 0;
    size_t down_off = 0, bytes = 0, stride = 0;
};
inline bool expert_fmt(int gu_type, int d_type, int n_embd, int n_ff, ExpertFmt& f) {
    if (!gate_up_type_ok(gu_type) || !down_type_ok(d_type)) return false;
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = row_bytes(gu_type, n_embd);
    f.d_row = row_bytes(d_type, n_ff);
    if (f.gu_row == 0 || f.d_row == 0) return false;
    f.gu_bytes = (size_t) 2 * n_ff * f.gu_row;
    f.d_bytes = (size_t) n_embd * f.d_row;
    f.down_off = (f.gu_bytes + 63) / 64 * 64;
    f.bytes = f.down_off + f.d_bytes;
    f.stride = (f.bytes + 63) / 64 * 64;
    return true;
}

/// The quantized activation of `n` values (n / 32 blocks), structure of arrays: q[n] int8, then s[n / 32] float, then
/// bsum[n / 32] int32, each part 64-byte aligned inside one allocation of act_bytes(n).
G4_HD constexpr size_t act_q_off(int64_t) { return 0; }
G4_HD constexpr size_t act_s_off(int64_t n) { return ((size_t) n + 63) / 64 * 64; }
G4_HD constexpr size_t act_b_off(int64_t n) { return act_s_off(n) + ((size_t) (n / 32) * 4 + 63) / 64 * 64; }
G4_HD constexpr size_t act_bytes(int64_t n) { return act_b_off(n) + ((size_t) (n / 32) * 4 + 63) / 64 * 64; }

}  // namespace strata::gemma4
