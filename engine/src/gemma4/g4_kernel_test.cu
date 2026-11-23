// src/gemma4/g4_kernel_test.cu - g4-kernel-test: the G4-A8 contract checked on real expert rows, and timed.
//
//   g4-kernel-test --model GGUF [--layer L] [--experts N] [--rows T] [--bench 1]
//
// Checks, bit for bit: the fp16 conversion, the activation quantizer, the gate/up rows, GeGLU + its activation and the
// down rows - scalar reference (g4_ref.hpp) against the AVX2 kernels (with and without AVX-VNNI) and the CUDA
// kernels.  Also reports how far the contract's result is from the dequantized weights applied to the unquantized
// activation (the error the 8-bit activations add), and with --bench 1 the throughput of each implementation and of a
// host-to-device copy of an expert.  Exit code 0 only when every bitwise check passes.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

#include <immintrin.h>

#include "g4_cpu.hpp"
#include "g4_expert.cuh"
#include "g4_fmt.hpp"
#include "g4_ref.hpp"
#include "strata/artifact/gguf_reader.hpp"

namespace g4 = strata::gemma4;

namespace {

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "g4-kernel-test: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int g_fail = 0;
void check(bool ok, const char* what, const std::string& detail = "") {
    std::printf("  %-58s %s%s%s\n", what, ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}
size_t diff_bytes(const void* a, const void* b, size_t n) {
    size_t d = 0;
    const uint8_t *x = (const uint8_t*) a, *y = (const uint8_t*) b;
    for (size_t i = 0; i < n; ++i) d += x[i] != y[i];
    return d;
}

// ---- dequantized rows, for the accuracy report (ggml's dequantize_row_* arithmetic in double)
void dequant_row(int type, const uint8_t* row, int n, double* y) {
    using g4::ref::fp16_to_fp32;
    if (type == g4::kTypeQ4_K || type == g4::kTypeQ5_K) {
        const bool q5 = type == g4::kTypeQ5_K;
        const size_t blk = q5 ? sizeof(g4::BlockQ5K) : sizeof(g4::BlockQ4K);
        for (int B = 0; B < n / 256; ++B) {
            const uint8_t* b = row + (size_t) B * blk;
            uint16_t hd, hm;
            std::memcpy(&hd, b, 2);
            std::memcpy(&hm, b + 2, 2);
            const double d = fp16_to_fp32(hd), dmin = fp16_to_fp32(hm);
            const uint8_t* qh = b + 16;
            const uint8_t* qs = b + 16 + (q5 ? 32 : 0);
            for (int j = 0; j < 8; ++j) {
                int sc, m;
                g4::k4_scale_min(j, b + 4, sc, m);
                for (int i = 0; i < 32; ++i) {
                    const uint8_t v = qs[(j / 2) * 32 + i];
                    int q = (j & 1) ? v >> 4 : v & 0xF;
                    if (q5) q += 16 * ((qh[i] >> j) & 1);
                    y[B * 256 + j * 32 + i] = d * sc * q - dmin * m;
                }
            }
        }
    } else if (type == g4::kTypeQ5_1) {
        const g4::BlockQ51* w = (const g4::BlockQ51*) row;
        for (int b = 0; b < n / 32; ++b) {
            const double d = fp16_to_fp32(w[b].d), m = fp16_to_fp32(w[b].m);
            uint32_t qh;
            std::memcpy(&qh, w[b].qh, 4);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = ((w[b].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) * d + m;
                y[b * 32 + j + 16] = ((w[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) * d + m;
            }
        }
    } else if (type == g4::kTypeIQ3_XXS) {   // ggml's dequantize_row_iq3_xxs
        for (int B = 0; B < n / 256; ++B) {
            const uint8_t* b = row + (size_t) B * 98;
            uint16_t hd;
            std::memcpy(&hd, b, 2);
            const double d = fp16_to_fp32(hd);
            const uint8_t* qs = b + 2;
            const uint8_t* sas = qs + 64;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux32;
                std::memcpy(&aux32, sas + 4 * ib, 4);
                const double db = d * (0.5 + (aux32 >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = g4::kIqSigns[(aux32 >> 7 * l) & 127];
                    const uint8_t* grid1 = (const uint8_t*) (g4::kIq3xxsGrid + qs[2 * l + 0]);
                    const uint8_t* grid2 = (const uint8_t*) (g4::kIq3xxsGrid + qs[2 * l + 1]);
                    for (int j = 0; j < 4; ++j) {
                        y[B * 256 + ib * 32 + l * 8 + j] = db * grid1[j] * ((signs >> j) & 1 ? -1.0 : 1.0);
                        y[B * 256 + ib * 32 + l * 8 + j + 4] = db * grid2[j] * ((signs >> (j + 4)) & 1 ? -1.0 : 1.0);
                    }
                }
                qs += 8;
            }
        }
    } else if (type == g4::kTypeIQ4_XS) {    // ggml's dequantize_row_iq4_xs
        for (int B = 0; B < n / 256; ++B) {
            const uint8_t* b = row + (size_t) B * 136;
            uint16_t hd, sh;
            std::memcpy(&hd, b, 2);
            std::memcpy(&sh, b + 2, 2);
            const double d = fp16_to_fp32(hd);
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                const double dl = d * (ls - 32);
                const uint8_t* qs = b + 8 + 16 * ib;
                for (int j = 0; j < 16; ++j) {
                    y[B * 256 + ib * 32 + j] = dl * g4::kIq4nl[qs[j] & 0xf];
                    y[B * 256 + ib * 32 + j + 16] = dl * g4::kIq4nl[qs[j] >> 4];
                }
            }
        }
    } else if (type == g4::kTypeIQ4_NL) {    // ggml's dequantize_row_iq4_nl
        for (int b = 0; b < n / 32; ++b) {
            const uint8_t* w = row + (size_t) b * 18;
            uint16_t hd;
            std::memcpy(&hd, w, 2);
            const double d = fp16_to_fp32(hd);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = d * g4::kIq4nl[w[2 + j] & 0xf];
                y[b * 32 + j + 16] = d * g4::kIq4nl[w[2 + j] >> 4];
            }
        }
    } else {
        const g4::BlockQ80* w = (const g4::BlockQ80*) row;
        for (int b = 0; b < n / 32; ++b) {
            const double d = fp16_to_fp32(w[b].d);
            for (int i = 0; i < 32; ++i) y[b * 32 + i] = d * w[b].qs[i];
        }
    }
}

template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model;
    int layer = 0, n_exp = 4, T = 4;
    bool bench = false;
    int mt = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--layer") layer = std::atoi(v.c_str());
        else if (a == "--experts") n_exp = std::atoi(v.c_str());
        else if (a == "--rows") T = std::atoi(v.c_str());
        else if (a == "--bench") bench = std::atoi(v.c_str()) != 0;
        else if (a == "--mt") mt = std::atoi(v.c_str());
        else {
            std::fprintf(stderr, "g4-kernel-test: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty() || T < 1 || T > g4::gpu::kMaxTok || n_exp < 1) {
        std::fprintf(stderr, "usage: g4-kernel-test --model GGUF [--layer L] [--experts N] [--rows 1..8] [--bench 1]\n");
        return 2;
    }
    if (!g4::cpu::available()) {
        std::fprintf(stderr, "g4-kernel-test: this CPU has no AVX2 + FMA + F16C\n");
        return 1;
    }
    strata::GgufModel gm = strata::GgufModel::open(model);
    const std::string pre = "blk." + std::to_string(layer) + ".";
    size_t sg = 0, sd = 0;
    const strata::TensorInfo* tg = gm.find(pre + "ffn_gate_up_exps.weight", &sg);
    const strata::TensorInfo* td = gm.find(pre + "ffn_down_exps.weight", &sd);
    if (!tg || !td || tg->shape.size() != 3 || td->shape.size() != 3) {
        std::fprintf(stderr, "g4-kernel-test: layer %d has no fused gate/up and down expert tensors\n", layer);
        return 1;
    }
    const int n_embd = (int) tg->shape[0], n_ff = (int) (tg->shape[1] / 2), n_expert = (int) tg->shape[2];
    g4::ExpertFmt f;
    if (!g4::expert_fmt((int) tg->type, (int) td->type, n_embd, n_ff, f) || (int) td->shape[0] != n_ff ||
        (int) td->shape[1] != n_embd) {
        std::fprintf(stderr, "g4-kernel-test: unsupported expert types %s / %s or shapes\n", tg->type_name(), td->type_name());
        return 1;
    }
    n_exp = std::min(n_exp, n_expert);
    std::printf("layer %d: %d experts, gate/up %s [%d x %d], down %s [%d x %d], blob %.2f MB; testing %d experts, %d rows; "
                "AVX-VNNI %s\n", layer, n_expert, tg->type_name(), 2 * n_ff, n_embd, td->type_name(), n_embd, n_ff,
                f.bytes / 1e6, n_exp, T, g4::cpu::have_avx_vnni() ? "yes" : "no");

    // host blobs (pinned) of the tested experts, spread over the layer
    uint8_t* hblob = nullptr;
    CK(cudaHostAlloc((void**) &hblob, f.stride * (size_t) n_exp, cudaHostAllocDefault));
    std::memset(hblob, 0, f.stride * (size_t) n_exp);
    std::vector<int> eid((size_t) n_exp);
    const uint8_t* src_g = gm.shard(sg).tensor_data(*tg);
    const uint8_t* src_d = gm.shard(sd).tensor_data(*td);
    for (int i = 0; i < n_exp; ++i) {
        eid[(size_t) i] = (int) ((int64_t) i * n_expert / n_exp);
        std::memcpy(hblob + (size_t) i * f.stride, src_g + (size_t) eid[(size_t) i] * f.gu_bytes, f.gu_bytes);
        std::memcpy(hblob + (size_t) i * f.stride + f.down_off, src_d + (size_t) eid[(size_t) i] * f.d_bytes, f.d_bytes);
    }

    // inputs: T rows with different statistics (unit normal, a wide dynamic range per block, a zero block, tiny values)
    std::mt19937_64 rng(20261122);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * n_embd);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_embd; ++i) {
            float v = nd(rng);
            if (t % 4 == 1) v *= std::exp(2.0f * nd(rng));
            if (t % 4 == 2 && i / 32 == 5) v = 0.0f;
            if (t % 4 == 3) v *= 1e-3f * (float) (1 + i % 7);
            x[(size_t) t * n_embd + i] = v;
        }

    std::printf("\nfp16 conversion\n");
    {
        size_t bad = 0, n = 0;
        auto one = [&](float v) {
            const uint16_t a = g4::ref::fp32_to_fp16(v);
            const uint16_t b = (uint16_t) _mm_cvtsi128_si32(
                _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT));
            const bool nan = v != v;
            if (nan ? ((a & 0x7c00) != 0x7c00 || (a & 0x3ff) == 0) : a != b) ++bad;
            ++n;
        };
        for (uint32_t h = 0; h < 65536; ++h) {   // every half, and the floats one ulp around it
            const float v = g4::ref::fp16_to_fp32((uint16_t) h);
            one(v);
            if (v == v) {
                one(std::nextafter(v, 1e30f));
                one(std::nextafter(v, -1e30f));
            }
        }
        std::uniform_int_distribution<uint32_t> ud;
        for (int i = 0; i < 4000000; ++i) one(g4::ref::f32_from_bits(ud(rng)));
        for (uint32_t h = 0; h + 1 < 65536; ++h) {   // the midpoints between adjacent halves (ties)
            const float a = g4::ref::fp16_to_fp32((uint16_t) h), b = g4::ref::fp16_to_fp32((uint16_t) (h + 1));
            if (a == a && b == b && std::isfinite(a) && std::isfinite(b)) one((float) (0.5 * ((double) a + (double) b)));
        }
        check(bad == 0, "reference fp32 -> fp16 equals F16C", std::to_string(bad) + " of " + std::to_string(n) + " differ");
    }

    if (mt > 0) {
        // --mt N: N threads, each running whole experts (one row, then `T` rows) over all of the layer's experts in turn,
        // straight from the mapped file's tensors copied to an unpinned arena: the CPU path's throughput ceiling
        const size_t all = f.stride * (size_t) n_expert;
        uint8_t* arena = (uint8_t*) _aligned_malloc(all, 4096);
        for (int e = 0; e < n_expert; ++e) {
            std::memcpy(arena + (size_t) e * f.stride, src_g + (size_t) e * f.gu_bytes, f.gu_bytes);
            std::memcpy(arena + (size_t) e * f.stride + f.down_off, src_d + (size_t) e * f.d_bytes, f.d_bytes);
        }
        std::vector<uint8_t> ax(g4::act_bytes(n_embd) * (size_t) T);
        for (int t = 0; t < T; ++t) g4::cpu::quantize_act(x.data() + (size_t) t * n_embd, n_embd, ax.data() + (size_t) t * g4::act_bytes(n_embd));
        std::printf("\nmulti-thread throughput over all %d experts of the layer (%.0f MB), whole experts, one per task\n", n_expert, all / 1e6);
        for (int nt : {1, T}) {
            std::vector<int> counts;
            for (int n = 1; n <= mt; n = n < 4 ? n + 1 : n + 4) counts.push_back(n);
            if (counts.back() != mt) counts.push_back(mt);
            for (int n : counts) {
                std::atomic<int> next{0};
                const int passes = 3;
                const auto t0 = Clock::now();
                std::vector<std::thread> th;
                for (int w = 0; w < n; ++w)
                    th.emplace_back([&, w] {
                        g4::cpu::Scratch scr;
                        std::vector<float> out((size_t) nt * n_embd);
                        const void* act[g4::cpu::kMaxTok];
                        float* o[g4::cpu::kMaxTok];
                        for (int t = 0; t < nt; ++t) {
                            act[t] = ax.data() + (size_t) t * g4::act_bytes(n_embd);
                            o[t] = out.data() + (size_t) t * n_embd;
                        }
                        for (;;) {
                            const int i = next.fetch_add(1);
                            if (i >= passes * n_expert) break;
                            g4::cpu::expert(f, arena + (size_t) (i % n_expert) * f.stride, act, nt, o, scr);
                        }
                    });
                for (auto& t : th) t.join();
                const double ms = ms_since(t0);
                std::printf("  %2d thread(s), %d row(s): %.1f GB/s of expert bytes (%.3f ms per expert per thread)\n", n, nt,
                            (double) passes * all / 1e6 / ms, ms * n / (passes * n_expert));
            }
        }
        _aligned_free(arena);
        return 0;
    }

    // ---- device state
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    const size_t ax_stride = g4::act_bytes(n_embd), ah_stride = g4::act_bytes(n_ff);
    const int K = n_exp;                    // entries per row in this test: every tested expert on every row
    const int n_ent = T * K;
    uint8_t* d_blob = dalloc<uint8_t>(f.stride * (size_t) n_exp);
    float* d_x = dalloc<float>((size_t) T * n_embd);
    uint8_t* d_ax = dalloc<uint8_t>(ax_stride * (size_t) T);
    uint8_t* d_ah = dalloc<uint8_t>(ah_stride * (size_t) n_ent);
    float* d_gu = dalloc<float>((size_t) n_ent * 2 * n_ff);
    float* d_parts = dalloc<float>((size_t) n_ent * n_embd);
    float* d_tab = dalloc<float>(65536);
    int32_t* d_ents = dalloc<int32_t>((size_t) n_ent);
    g4::gpu::Group* d_groups = dalloc<g4::gpu::Group>((size_t) n_exp);
    CK(cudaMemcpy(d_blob, hblob, f.stride * (size_t) n_exp, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tab, g4::ref::gelu_table(), 65536 * 4, cudaMemcpyHostToDevice));
    std::vector<g4::gpu::Group> groups((size_t) n_exp);
    std::vector<int32_t> ents((size_t) n_ent);
    for (int e = 0; e < n_exp; ++e) {
        g4::gpu::Group& G = groups[(size_t) e];
        G.blob = d_blob + (size_t) e * f.stride;
        G.nt = T;
        for (int t = 0; t < T; ++t) {
            G.ent[t] = t * K + e;
            G.row[t] = t;
        }
    }
    for (int i = 0; i < n_ent; ++i) ents[(size_t) i] = i;
    CK(cudaMemcpy(d_groups, groups.data(), groups.size() * sizeof(g4::gpu::Group), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ents, ents.data(), ents.size() * 4, cudaMemcpyHostToDevice));
    const g4::gpu::ActDev dax{d_ax, ax_stride, n_embd}, dah{d_ah, ah_stride, n_ff};
    auto gpu_all = [&]() {
        g4::gpu::k_quant_a8<<<dim3((unsigned) (n_embd / 32), (unsigned) T), 32, 0, st>>>(d_x, n_embd, dax);
        g4::gpu::launch_gate_up(f, d_groups, n_exp, dax, d_gu, st);
        g4::gpu::k_geglu_quant<<<dim3((unsigned) (n_ff / 32), (unsigned) n_ent), 32, 0, st>>>(d_ents, d_gu, d_tab, dah, n_ff);
        g4::gpu::launch_down(f, d_groups, n_exp, dah, d_parts, st);
    };
    gpu_all();
    CK(cudaStreamSynchronize(st));
    std::vector<uint8_t> g_ax(ax_stride * (size_t) T), g_ah(ah_stride * (size_t) n_ent);
    std::vector<float> g_gu((size_t) n_ent * 2 * n_ff), g_parts((size_t) n_ent * n_embd);
    CK(cudaMemcpy(g_ax.data(), d_ax, g_ax.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_ah.data(), d_ah, g_ah.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_gu.data(), d_gu, g_gu.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_parts.data(), d_parts, g_parts.size() * 4, cudaMemcpyDeviceToHost));

    // ---- reference and AVX2
    std::vector<uint8_t> r_ax(ax_stride * (size_t) T, 0), c_ax(ax_stride * (size_t) T, 0);
    for (int t = 0; t < T; ++t) {
        g4::ref::quantize_act(x.data() + (size_t) t * n_embd, n_embd, r_ax.data() + (size_t) t * ax_stride);
        g4::cpu::quantize_act(x.data() + (size_t) t * n_embd, n_embd, c_ax.data() + (size_t) t * ax_stride);
    }
    // compare only the meaningful bytes of an activation (the padding between its parts is not written)
    auto act_diff = [&](const uint8_t* a, const uint8_t* b, int n) {
        return diff_bytes(a, b, (size_t) n) + diff_bytes(a + g4::act_s_off(n), b + g4::act_s_off(n), (size_t) (n / 32) * 4) +
               diff_bytes(a + g4::act_b_off(n), b + g4::act_b_off(n), (size_t) (n / 32) * 4);
    };
    std::printf("\nactivation quantizer (%d rows of %d)\n", T, n_embd);
    {
        size_t dc = 0, dg = 0;
        for (int t = 0; t < T; ++t) {
            dc += act_diff(r_ax.data() + (size_t) t * ax_stride, c_ax.data() + (size_t) t * ax_stride, n_embd);
            dg += act_diff(r_ax.data() + (size_t) t * ax_stride, g_ax.data() + (size_t) t * ax_stride, n_embd);
        }
        check(dc == 0, "AVX2 equals the reference", std::to_string(dc) + " bytes differ");
        check(dg == 0, "CUDA equals the reference", std::to_string(dg) + " bytes differ");
    }

    std::vector<float> r_gu((size_t) n_ent * 2 * n_ff), r_h((size_t) n_ent * n_ff), r_parts((size_t) n_ent * n_embd);
    std::vector<uint8_t> r_ah(ah_stride * (size_t) n_ent, 0);
    const auto t_ref = Clock::now();
    for (int e = 0; e < n_exp; ++e)
        for (int t = 0; t < T; ++t) {
            const int ent = t * K + e;
            const uint8_t* blob = hblob + (size_t) e * f.stride;
            const g4::ref::Act ax = g4::ref::act_view(r_ax.data() + (size_t) t * ax_stride, n_embd);
            float* gu = r_gu.data() + (size_t) ent * 2 * n_ff;
            for (int r = 0; r < 2 * n_ff; ++r) gu[r] = g4::ref::dot(f.gu_type, blob + (size_t) r * f.gu_row, n_embd, ax);
            g4::ref::geglu(gu, n_ff, r_h.data() + (size_t) ent * n_ff);
            g4::ref::quantize_act(r_h.data() + (size_t) ent * n_ff, n_ff, r_ah.data() + (size_t) ent * ah_stride);
            const g4::ref::Act ah = g4::ref::act_view(r_ah.data() + (size_t) ent * ah_stride, n_ff);
            float* out = r_parts.data() + (size_t) ent * n_embd;
            for (int r = 0; r < n_embd; ++r) out[r] = g4::ref::dot(f.d_type, blob + f.down_off + (size_t) r * f.d_row, n_ff, ah);
        }
    const double ref_ms = ms_since(t_ref);

    // AVX2: every expert for all T rows at once (the multi-token path), then row by row (the single-token path)
    std::vector<float> c_parts((size_t) n_ent * n_embd), c1_parts((size_t) n_ent * n_embd);
    std::vector<float> c_gu((size_t) n_ent * 2 * n_ff);
    std::vector<uint8_t> c_ah(ah_stride * (size_t) n_ent, 0);
    std::vector<float> c_h((size_t) n_ff);
    g4::cpu::Scratch scr;
    for (int e = 0; e < n_exp; ++e) {
        const uint8_t* blob = hblob + (size_t) e * f.stride;
        const void* act[g4::cpu::kMaxTok];
        float* out[g4::cpu::kMaxTok];
        float* gu[g4::cpu::kMaxTok];
        const void* ah[g4::cpu::kMaxTok];
        for (int t = 0; t < T; ++t) {
            act[t] = r_ax.data() + (size_t) t * ax_stride;
            out[t] = c_parts.data() + (size_t) (t * K + e) * n_embd;
            gu[t] = c_gu.data() + (size_t) (t * K + e) * 2 * n_ff;
            ah[t] = c_ah.data() + (size_t) (t * K + e) * ah_stride;
        }
        g4::cpu::gate_up_rows(f, blob, act, T, gu, 0, 2 * n_ff);
        for (int t = 0; t < T; ++t) g4::cpu::geglu_quantize(gu[t], n_ff, c_h.data(), const_cast<void*>(ah[t]));
        g4::cpu::down_rows(f, blob, ah, T, out, 0, n_embd);
        for (int t = 0; t < T; ++t) {
            const void* a1[1] = {act[t]};
            float* o1[1] = {c1_parts.data() + (size_t) (t * K + e) * n_embd};
            g4::cpu::expert(f, blob, a1, 1, o1, scr);
        }
    }
    std::printf("\ngate/up rows (%d entries of %d)\n", n_ent, 2 * n_ff);
    check(diff_bytes(r_gu.data(), c_gu.data(), r_gu.size() * 4) == 0, "AVX2 equals the reference",
          std::to_string(diff_bytes(r_gu.data(), c_gu.data(), r_gu.size() * 4)) + " bytes differ");
    check(diff_bytes(r_gu.data(), g_gu.data(), r_gu.size() * 4) == 0, "CUDA equals the reference",
          std::to_string(diff_bytes(r_gu.data(), g_gu.data(), r_gu.size() * 4)) + " bytes differ");
    std::printf("\nGeGLU and its activation\n");
    {
        size_t dc = 0, dg = 0;
        for (int i = 0; i < n_ent; ++i) {
            dc += act_diff(r_ah.data() + (size_t) i * ah_stride, c_ah.data() + (size_t) i * ah_stride, n_ff);
            dg += act_diff(r_ah.data() + (size_t) i * ah_stride, g_ah.data() + (size_t) i * ah_stride, n_ff);
        }
        check(dc == 0, "AVX2 equals the reference", std::to_string(dc) + " bytes differ");
        check(dg == 0, "CUDA equals the reference", std::to_string(dg) + " bytes differ");
    }
    std::printf("\ndown rows (%d entries of %d)\n", n_ent, n_embd);
    check(diff_bytes(r_parts.data(), c_parts.data(), r_parts.size() * 4) == 0, "AVX2 (all rows of a group) equals the reference",
          std::to_string(diff_bytes(r_parts.data(), c_parts.data(), r_parts.size() * 4)) + " bytes differ");
    check(diff_bytes(r_parts.data(), c1_parts.data(), r_parts.size() * 4) == 0, "AVX2 (one row at a time) equals the reference",
          std::to_string(diff_bytes(r_parts.data(), c1_parts.data(), r_parts.size() * 4)) + " bytes differ");
    check(diff_bytes(r_parts.data(), g_parts.data(), r_parts.size() * 4) == 0, "CUDA equals the reference",
          std::to_string(diff_bytes(r_parts.data(), g_parts.data(), r_parts.size() * 4)) + " bytes differ");

    // ---- how far is the contract from the dequantized weights on the unquantized activation?
    {
        std::vector<double> wrow((size_t) std::max(n_embd, n_ff)), gu((size_t) 2 * n_ff), h((size_t) n_ff);
        double num = 0, den = 0, num_gu = 0, den_gu = 0;
        const double c = std::sqrt(2.0 / 3.14159265358979323846);
        const int e = 0;
        for (int t = 0; t < T; ++t) {
            const uint8_t* blob = hblob + (size_t) e * f.stride;
            const float* xt = x.data() + (size_t) t * n_embd;
            for (int r = 0; r < 2 * n_ff; ++r) {
                dequant_row(f.gu_type, blob + (size_t) r * f.gu_row, n_embd, wrow.data());
                double s = 0;
                for (int i = 0; i < n_embd; ++i) s += wrow[(size_t) i] * xt[i];
                gu[(size_t) r] = s;
                const double dlt = s - r_gu[(size_t) (t * K + e) * 2 * n_ff + r];
                num_gu += dlt * dlt;
                den_gu += s * s;
            }
            for (int j = 0; j < n_ff; ++j) {
                const double g = gu[(size_t) j];
                h[(size_t) j] = 0.5 * g * (1.0 + std::tanh(c * (g + 0.044715 * g * g * g))) * gu[(size_t) (n_ff + j)];
            }
            for (int r = 0; r < n_embd; ++r) {
                dequant_row(f.d_type, blob + f.down_off + (size_t) r * f.d_row, n_ff, wrow.data());
                double s = 0;
                for (int j = 0; j < n_ff; ++j) s += wrow[(size_t) j] * h[(size_t) j];
                const double dlt = s - r_parts[(size_t) (t * K + e) * n_embd + r];
                num += dlt * dlt;
                den += s * s;
            }
        }
        std::printf("\naccuracy of the contract against dequantized weights x unquantized activations (expert %d, %d rows)\n", eid[0], T);
        std::printf("  gate/up outputs: relative L2 error %.3e\n  expert outputs:  relative L2 error %.3e\n",
                    std::sqrt(num_gu / std::max(den_gu, 1e-300)), std::sqrt(num / std::max(den, 1e-300)));
    }

    if (bench) {
        std::printf("\nthroughput (expert blob %.2f MB; one thread on the CPU)\n", f.bytes / 1e6);
        std::printf("  scalar reference: %.1f ms per expert-row\n", ref_ms / n_ent);
        for (int nt : {1, T}) {
            const void* act[g4::cpu::kMaxTok];
            float* out[g4::cpu::kMaxTok];
            for (int t = 0; t < nt; ++t) {
                act[t] = r_ax.data() + (size_t) t * ax_stride;
                out[t] = c_parts.data() + (size_t) t * n_embd;
            }
            const int reps = 40;
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                const auto t0 = Clock::now();
                for (int i = 0; i < reps; ++i) g4::cpu::expert(f, hblob + (size_t) (i % n_exp) * f.stride, act, nt, out, scr);
                best = std::min(best, ms_since(t0) / reps);
            }
            std::printf("  AVX2 whole expert, %d row(s): %.3f ms (%.1f GB/s of expert bytes, %.3f ms per row)\n", nt, best,
                        f.bytes / 1e6 / best, best / nt);
        }
        {
            const int reps = 200;
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                CK(cudaStreamSynchronize(st));
                const auto t0 = Clock::now();
                for (int i = 0; i < reps; ++i) gpu_all();
                CK(cudaStreamSynchronize(st));
                best = std::min(best, ms_since(t0) / reps);
            }
            std::printf("  CUDA %d experts x %d rows (quantize + gate/up + geglu + down): %.3f ms (%.1f GB/s of expert bytes)\n",
                        n_exp, T, best, f.bytes * n_exp / 1e6 / best);
        }
        {
            const int reps = 50;
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                CK(cudaStreamSynchronize(st));
                const auto t0 = Clock::now();
                for (int i = 0; i < reps; ++i)
                    CK(cudaMemcpyAsync(d_blob + (size_t) (i % n_exp) * f.stride, hblob + (size_t) (i % n_exp) * f.stride, f.bytes,
                                       cudaMemcpyHostToDevice, st));
                CK(cudaStreamSynchronize(st));
                best = std::min(best, ms_since(t0) / reps);
            }
            std::printf("  pinned host -> device copy of one expert: %.3f ms (%.1f GB/s)\n", best, f.bytes / 1e6 / best);
        }
    }
    std::printf("\n%s\n", g_fail == 0 ? "ALL BITWISE CHECKS PASSED" : "BITWISE CHECKS FAILED");
    return g_fail == 0 ? 0 : 1;
}
