// src/gemma4/g4_cpu.hpp - the CPU side of the Gemma 4 routed experts (AVX2 + FMA + F16C), bitwise the scalar
// reference of g4_ref.hpp (the G4-A8 contract, g4_fmt.hpp).
#pragma once

#include <cstddef>
#include <cstdint>

#include "g4_fmt.hpp"

namespace strata::gemma4::cpu {

/// What the kernels were built for and what this CPU has.
bool available() noexcept;          // AVX2 + FMA + F16C
bool have_avx_vnni() noexcept;      // VPDPBUSD (VEX), used when present (the same integers)

/// x (n floats, n % 32 == 0) -> its A8 activation (act_bytes(n) bytes, 64-byte aligned).
void quantize_act(const float* x, int64_t n, void* act);

constexpr int kMaxTok = 8;   // rows of a decode / verify window

/// Gate/up rows [r0, r1) of one expert (rows of f.n_embd weights, f.gu_type) against the A8 activations of `nt`
/// tokens: out[t][r] for r in [r0, r1).  The weight row is unpacked once for all tokens.
void gate_up_rows(const ExpertFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* out, int r0, int r1);

/// h[j] = gelu_tab[fp16(gate_j)] * up_j for j in [0, n_ff), then its A8 activation into `act_h` (act_bytes(n_ff)).
void geglu_quantize(const float* gate_up, int n_ff, float* h, void* act_h);

/// Down rows [r0, r1) (rows of f.n_ff weights, f.d_type) against the A8 activations of the tokens' hidden vectors:
/// out[t][r] for r in [r0, r1).
void down_rows(const ExpertFmt& f, const uint8_t* blob, const void* const* act_h, int nt, float* const* out, int r0, int r1);

/// Per-thread scratch of whole-expert calls.
struct Scratch {
    float* gu = nullptr;      // [kMaxTok][2 * n_ff]
    float* h = nullptr;       // [n_ff]
    uint8_t* act_h = nullptr; // [kMaxTok][act_bytes(n_ff)]
    int n_ff = 0;
    Scratch() = default;
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    ~Scratch();
    void reserve(int n_ff);
};

/// One whole expert for `nt` tokens: out[t][n_embd].
void expert(const ExpertFmt& f, const uint8_t* blob, const void* const* act_x, int nt, float* const* out, Scratch& s);

}  // namespace strata::gemma4::cpu
