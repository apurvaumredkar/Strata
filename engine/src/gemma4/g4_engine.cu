// src/gemma4/g4_engine.cu - strata-g4: the engine path for Gemma 4 26B A4B (GGUF architecture gemma4).
//
// The model (30 layers): attention (25 sliding-window layers of 16 query / 8 KV heads of 256, window 1024; 5 global
// layers of 16 / 2 heads of 512 whose values are their keys' projection), then a dense GeGLU FFN and a 128-expert
// top-8 MoE side by side, each with its own norms.  The head is the token embedding (262,144 x 2816).
//
// How it runs on a card that cannot hold the experts (13.5 of the file's 15.9 GiB):
//   * the dense weights stay in VRAM in their GGUF blocks (strata's native MMVQ); the attention, the router and the
//     norms are this path's own kernels (g4_kernels.cuh);
//   * every expert sits in pinned host memory as a blob [gate_up | down]; VRAM left over holds `slots` of them per
//     layer.  A routed expert with a slot runs on the GPU (g4_expert.cuh), one without on the CPU pool (g4_cpu.cpp),
//     side by side with the GPU's dense FFN.  Both give the same bits (g4_fmt.hpp), so the text does not depend on
//     what the cache holds;
//   * the cache follows the routing while a request runs: the host counts every layer's picks (it reads them back
//     anyway) and copies the experts in demand into the least-used slots on a second stream.  An expert being copied
//     counts as absent until its copy has finished.
//
// Modes: --tokens-file F [--n N] generates greedily and prints "prompt:" / "gen:" lines (the interface of
// tools/oracle/g4-oracle, so the two can be compared), with --top2 / --full-logits dumps and --force-file teacher
// forcing; --jobs-file F [--reps R] runs several requests one after the other in one process (the expert cache
// carries over, as in a server); --bench-json F writes every request's timings and tokens.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/gemma4/engine.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "g4_cpu.hpp"
#include "g4_expert.cuh"
#include "g4_fmt.hpp"
#include "g4_kernels.cuh"
#include "g4_ref.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <immintrin.h>

namespace {

namespace K = strata::kernels;
namespace C = strata::kernels::cpu;
namespace g4 = strata::gemma4;
namespace G = strata::gemma4::gpu;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-g4: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "strata-g4: %s\n", msg.c_str());
    std::exit(1);
}

constexpr int kTmax = 8;   // rows of a forward (strata's MMVQ takes 1..8 columns; an expert group 1..8 entries)

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1, n_in = 0, n_out = 0;
};

struct Layer {
    bool swa = true;
    int H = 16, HKV = 8, HD = 256;
    float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *post_attn_norm = nullptr, *ffn_norm = nullptr,
          *post_ffw_norm_1 = nullptr, *pre_ffw_norm_2 = nullptr, *post_ffw_norm_2 = nullptr, *post_ffw_norm = nullptr,
          *router_scale = nullptr, *router = nullptr, *exp_scale = nullptr;
    float out_scale = 1.0f;
    Mat wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down;
    bool has_v = false;
    // the routed experts
    g4::ExpertFmt f;
    uint8_t* hblobs = nullptr;   // host: every expert, f.stride apart
    uint8_t* slots = nullptr;    // device: n_slots blobs
    int n_slots = 0;
    // the attention cache: [HKV][cap][HD] fp16
    __half *kc = nullptr, *vc = nullptr;
    int cap = 0;
    // the expert cache's tables (host)
    std::vector<int16_t> slot_of;       // expert -> its slot, -1: none
    std::vector<int16_t> exp_in_slot;   // slot -> its expert, -1: empty or being filled
    std::vector<float> score;           // decayed use counts
    std::vector<std::pair<int, int>> pending;   // (expert, slot) copies in flight
    cudaEvent_t ev_up = nullptr, ev_exp = nullptr;
    bool up_busy = false;
};

struct Model {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_expert = 0, top_k = 0, n_ff_exp = 0, n_vocab = 0, n_swa = 0;
    float eps = 1e-6f, softcap = 0.0f, rope_base = 1e6f, rope_base_swa = 1e4f;
    int hd = 512, hd_swa = 256;
    std::vector<Layer> layers;
    float *out_norm = nullptr, *rope_freqs = nullptr;
    Mat embd;   // token_embd: the embedding rows and the head
    bool pinned = false;
    size_t expert_bytes = 0, dense_bytes = 0;
};

float* upload_f32(const strata::GgufModel& gm, const std::string& name, uint64_t want, std::vector<float>* host = nullptr) {
    size_t s = 0;
    const strata::TensorInfo* t = gm.find(name, &s);
    if (!t) die("missing tensor " + name);
    if (t->type != 0) die(name + " is " + t->type_name() + ", expected F32");
    if (want && t->elements() != want) die(name + " has " + std::to_string(t->elements()) + " values, expected " + std::to_string(want));
    const float* src = (const float*) gm.shard(s).tensor_data(*t);
    if (host) host->assign(src, src + t->elements());
    float* d = nullptr;
    CK(cudaMalloc(&d, t->elements() * 4));
    CK(cudaMemcpy(d, src, t->elements() * 4, cudaMemcpyHostToDevice));
    return d;
}

Mat upload_mat(const strata::GgufModel& gm, const std::string& name, int n_in, int n_out, size_t& bytes) {
    size_t s = 0;
    const strata::TensorInfo* t = gm.find(name, &s);
    if (!t) die("missing tensor " + name);
    if (t->shape.size() != 2 || (int) t->shape[0] != n_in || (int) t->shape[1] != n_out)
        die(name + " is not [" + std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
    if (!K::native_mmvq_supported((int) t->type)) die(name + " is " + t->type_name() + ", which the native MMVQ does not take");
    Mat w;
    w.type = (int) t->type;
    w.n_in = n_in;
    w.n_out = n_out;
    const uint64_t nb = strata::tensor_payload_bytes(*t);
    CK(cudaMalloc(&w.d, nb + 64));
    CK(cudaMemset(w.d + nb, 0, 64));
    CK(cudaMemcpy(w.d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
    bytes += nb;
    return w;
}

uint64_t meta_u(const strata::GgufModel& gm, const std::string& key) {
    const strata::MetaValue* v = gm.meta().get(key);
    if (!v || !v->is_num()) die("missing metadata " + key);
    return v->u;
}
double meta_f(const strata::GgufModel& gm, const std::string& key, double dflt) {
    const strata::MetaValue* v = gm.meta().get(key);
    return v && v->is_num() ? v->num() : dflt;
}

void load(const std::string& path, Model& m) {
    const auto t0 = Clock::now();
    strata::GgufModel gm = strata::GgufModel::open(path);
    const strata::MetaValue* arch = gm.meta().get("general.architecture");
    if (!arch || arch->s != "gemma4") die("this engine path runs gemma4 files; this one is '" + (arch ? arch->s : std::string("?")) + "'");
    const std::string p = "gemma4.";
    m.n_layer = (int) meta_u(gm, p + "block_count");
    m.n_embd = (int) meta_u(gm, p + "embedding_length");
    m.n_ff = (int) meta_u(gm, p + "feed_forward_length");
    m.n_expert = (int) meta_u(gm, p + "expert_count");
    m.top_k = (int) meta_u(gm, p + "expert_used_count");
    m.n_ff_exp = (int) meta_u(gm, p + "expert_feed_forward_length");
    m.n_swa = (int) meta_u(gm, p + "attention.sliding_window");
    m.hd = (int) meta_u(gm, p + "attention.key_length");
    m.hd_swa = (int) meta_u(gm, p + "attention.key_length_swa");
    m.eps = (float) meta_f(gm, p + "attention.layer_norm_rms_epsilon", 1e-6);
    m.softcap = (float) meta_f(gm, p + "final_logit_softcapping", 0.0);
    m.rope_base = (float) meta_f(gm, p + "rope.freq_base", 1e6);
    m.rope_base_swa = (float) meta_f(gm, p + "rope.freq_base_swa", 1e4);
    const int n_head = (int) meta_u(gm, p + "attention.head_count");
    const strata::MetaValue* hkv = gm.meta().get(p + "attention.head_count_kv");
    const strata::MetaValue* swp = gm.meta().get(p + "attention.sliding_window_pattern");
    if (!hkv || (int) hkv->items.size() != m.n_layer || !swp || (int) swp->items.size() != m.n_layer)
        die("the per-layer head_count_kv / sliding_window_pattern arrays are missing");
    if (meta_f(gm, p + "embedding_length_per_layer_input", 0) != 0) die("per-layer input embeddings (E2B / E4B) are not implemented");
    if (meta_f(gm, p + "attention.shared_kv_layers", 0) != 0) die("shared-KV layers (E2B / E4B) are not implemented");
    if (m.n_embd % 256 || m.n_ff % 32 || m.n_ff_exp % 32 || m.n_expert > 256 || m.top_k > 16 || m.n_expert < m.top_k)
        die("a geometry this path's kernels do not take");
    if ((m.hd != 256 && m.hd != 512) || (m.hd_swa != 256 && m.hd_swa != 512)) die("head widths other than 256 / 512 are not implemented");
    {
        size_t s = 0;
        const strata::TensorInfo* t = gm.find("token_embd.weight", &s);
        if (!t || t->type != 8 || (int) t->shape[0] != m.n_embd) die("token_embd.weight must be Q8_0 [n_embd, vocab]");
        m.n_vocab = (int) t->shape[1];
        if (gm.find("output.weight")) die("a separate output head is not implemented (Gemma 4 ties it to token_embd)");
    }
    size_t dense = 0;
    m.embd = upload_mat(gm, "token_embd.weight", m.n_embd, m.n_vocab, dense);
    m.out_norm = upload_f32(gm, "output_norm.weight", (uint64_t) m.n_embd);
    m.rope_freqs = upload_f32(gm, "rope_freqs.weight", (uint64_t) m.hd / 2);
    m.layers.resize((size_t) m.n_layer);
    m.pinned = true;
    for (int l = 0; l < m.n_layer; ++l) {
        Layer& y = m.layers[(size_t) l];
        auto nm = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
        y.swa = swp->items[(size_t) l].u != 0;
        y.H = n_head;
        y.HKV = (int) hkv->items[(size_t) l].u;
        y.HD = y.swa ? m.hd_swa : m.hd;
        if (y.H % y.HKV) die("head_count is not a multiple of head_count_kv");
        const uint64_t N = (uint64_t) m.n_embd;
        y.attn_norm = upload_f32(gm, nm("attn_norm.weight"), N);
        y.q_norm = upload_f32(gm, nm("attn_q_norm.weight"), (uint64_t) y.HD);
        y.k_norm = upload_f32(gm, nm("attn_k_norm.weight"), (uint64_t) y.HD);
        y.post_attn_norm = upload_f32(gm, nm("post_attention_norm.weight"), N);
        y.ffn_norm = upload_f32(gm, nm("ffn_norm.weight"), N);
        y.post_ffw_norm_1 = upload_f32(gm, nm("post_ffw_norm_1.weight"), N);
        y.pre_ffw_norm_2 = upload_f32(gm, nm("pre_ffw_norm_2.weight"), N);
        y.post_ffw_norm_2 = upload_f32(gm, nm("post_ffw_norm_2.weight"), N);
        y.post_ffw_norm = upload_f32(gm, nm("post_ffw_norm.weight"), N);
        y.router_scale = upload_f32(gm, nm("ffn_gate_inp.scale"), N);
        y.router = upload_f32(gm, nm("ffn_gate_inp.weight"), N * (uint64_t) m.n_expert);
        y.exp_scale = upload_f32(gm, nm("ffn_down_exps.scale"), (uint64_t) m.n_expert);
        {
            std::vector<float> os;
            float* d = upload_f32(gm, nm("layer_output_scale.weight"), 1, &os);
            CK(cudaFree(d));
            y.out_scale = os[0];
        }
        y.wq = upload_mat(gm, nm("attn_q.weight"), m.n_embd, y.H * y.HD, dense);
        y.wk = upload_mat(gm, nm("attn_k.weight"), m.n_embd, y.HKV * y.HD, dense);
        y.has_v = gm.find(nm("attn_v.weight")) != nullptr;
        if (y.has_v) y.wv = upload_mat(gm, nm("attn_v.weight"), m.n_embd, y.HKV * y.HD, dense);
        y.wo = upload_mat(gm, nm("attn_output.weight"), y.H * y.HD, m.n_embd, dense);
        y.ffn_gate = upload_mat(gm, nm("ffn_gate.weight"), m.n_embd, m.n_ff, dense);
        y.ffn_up = upload_mat(gm, nm("ffn_up.weight"), m.n_embd, m.n_ff, dense);
        y.ffn_down = upload_mat(gm, nm("ffn_down.weight"), m.n_ff, m.n_embd, dense);
        // the routed experts into host blobs [gate_up | down]
        size_t sg = 0, sd = 0;
        const strata::TensorInfo* tg = gm.find(nm("ffn_gate_up_exps.weight"), &sg);
        const strata::TensorInfo* td = gm.find(nm("ffn_down_exps.weight"), &sd);
        if (!tg || !td) die("layer " + std::to_string(l) + " has no fused gate/up and down expert tensors");
        if (tg->shape.size() != 3 || td->shape.size() != 3 || (int) tg->shape[0] != m.n_embd ||
            (int) tg->shape[1] != 2 * m.n_ff_exp || (int) tg->shape[2] != m.n_expert || (int) td->shape[0] != m.n_ff_exp ||
            (int) td->shape[1] != m.n_embd || (int) td->shape[2] != m.n_expert)
            die("layer " + std::to_string(l) + ": unexpected expert tensor shapes");
        if (!g4::expert_fmt((int) tg->type, (int) td->type, m.n_embd, m.n_ff_exp, y.f))
            die("layer " + std::to_string(l) + ": experts are " + tg->type_name() + " / " + td->type_name() +
                "; this path takes Q4_K or Q5_K gate/up and Q5_1 or Q8_0 down");
        const size_t total = y.f.stride * (size_t) m.n_expert;
        if (m.pinned && cudaHostAlloc((void**) &y.hblobs, total, cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            m.pinned = false;
        }
        if (!y.hblobs) {
            y.hblobs = (uint8_t*) _aligned_malloc(total, 4096);
            if (!y.hblobs) die("no host memory for the experts of layer " + std::to_string(l));
        }
        const uint8_t* src_g = gm.shard(sg).tensor_data(*tg);
        const uint8_t* src_d = gm.shard(sd).tensor_data(*td);
        for (int e = 0; e < m.n_expert; ++e) {
            uint8_t* b = y.hblobs + (size_t) e * y.f.stride;
            std::memcpy(b, src_g + (size_t) e * y.f.gu_bytes, y.f.gu_bytes);
            std::memset(b + y.f.gu_bytes, 0, y.f.down_off - y.f.gu_bytes);
            std::memcpy(b + y.f.down_off, src_d + (size_t) e * y.f.d_bytes, y.f.d_bytes);
            std::memset(b + y.f.bytes, 0, y.f.stride - y.f.bytes);
        }
        m.expert_bytes += total;
        y.slot_of.assign((size_t) m.n_expert, -1);
        y.score.assign((size_t) m.n_expert, 0.0f);
        CK(cudaEventCreateWithFlags(&y.ev_up, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&y.ev_exp, cudaEventDisableTiming));
    }
    m.dense_bytes = dense;
    std::fprintf(stderr, "strata-g4: %s: %d layers, %d experts (top-%d) of %d, dense %.2f GiB on the GPU, experts %.2f GiB in "
                 "%s host memory, %.1f s\n", path.c_str(), m.n_layer, m.n_expert, m.top_k, m.n_ff_exp,
                 (double) dense / (1 << 30), (double) m.expert_bytes / (1 << 30), m.pinned ? "pinned" : "pageable",
                 ms_since(t0) / 1000.0);
}

// ------------------------------------------------------------------------------------------------ the engine
template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T> T* halloc(size_t n) {
    T* p = nullptr;
    CK(cudaHostAlloc((void**) &p, n * sizeof(T), cudaHostAllocDefault));
    std::memset(p, 0, n * sizeof(T));
    return p;
}

struct CpuJob {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const void* act[kTmax];
    float* out[kTmax];
};
struct CpuCtx {
    const g4::ExpertFmt* f = nullptr;
    const CpuJob* jobs = nullptr;
};
void cpu_job_fn(void* vctx, int i) {
    static thread_local g4::cpu::Scratch scratch;
    const CpuCtx& c = *(const CpuCtx*) vctx;
    const CpuJob& j = c.jobs[i];
    g4::cpu::expert(*c.f, j.blob, j.act, j.nt, j.out, scratch);
}

/// A layer's missed experts split by rows over every pool thread (fewer experts than threads would leave cores idle
/// and stream each expert at one core's speed): gate/up row chunks, then GeGLU + its activation per expert, then down
/// row chunks.  Task i of a row phase is chunk i % chunks of job i / chunks.
struct SplitCtx {
    const g4::ExpertFmt* f = nullptr;
    const CpuJob* jobs = nullptr;
    int chunks = 1;
    float* gu = nullptr;        // [job][kTmax][2 * n_ff]
    uint8_t* ah = nullptr;      // [job][kTmax][act_bytes(n_ff)]
    size_t ah_stride = 0;
};
void split_gu_fn(void* vctx, int i) {
    const SplitCtx& c = *(const SplitCtx*) vctx;
    const int job = i / c.chunks, ch = i % c.chunks, rows = 2 * c.f->n_ff;
    const CpuJob& j = c.jobs[job];
    float* out[kTmax];
    for (int t = 0; t < j.nt; ++t) out[t] = c.gu + ((size_t) job * kTmax + t) * (size_t) rows;
    g4::cpu::gate_up_rows(*c.f, j.blob, j.act, j.nt, out, (int) ((int64_t) rows * ch / c.chunks), (int) ((int64_t) rows * (ch + 1) / c.chunks));
}
void split_mid_fn(void* vctx, int job) {
    static thread_local std::vector<float> h;
    const SplitCtx& c = *(const SplitCtx*) vctx;
    const CpuJob& j = c.jobs[job];
    h.resize((size_t) c.f->n_ff);
    for (int t = 0; t < j.nt; ++t)
        g4::cpu::geglu_quantize(c.gu + ((size_t) job * kTmax + t) * (size_t) (2 * c.f->n_ff), c.f->n_ff, h.data(),
                                c.ah + ((size_t) job * kTmax + t) * c.ah_stride);
}
void split_down_fn(void* vctx, int i) {
    const SplitCtx& c = *(const SplitCtx*) vctx;
    const int job = i / c.chunks, ch = i % c.chunks, rows = c.f->n_embd;
    const CpuJob& j = c.jobs[job];
    const void* ah[kTmax];
    for (int t = 0; t < j.nt; ++t) ah[t] = c.ah + ((size_t) job * kTmax + t) * c.ah_stride;
    g4::cpu::down_rows(*c.f, j.blob, ah, j.nt, j.out, (int) ((int64_t) rows * ch / c.chunks), (int) ((int64_t) rows * (ch + 1) / c.chunks));
}

struct Stats {
    long long passes = 0, rows = 0, ent_gpu = 0, ent_cpu = 0, uploads = 0, cpu_jobs = 0;
    double ms_cpu = 0, ms_wait_route = 0, ms_plan = 0;
    void reset() { *this = Stats(); }
};

struct Engine {
    Model* m = nullptr;
    cudaStream_t st = nullptr, up = nullptr;
    int ctx = 4096, ring = 0;
    int64_t pos = 0;   // cells holding a processed token
    // device buffers, [kTmax] rows each
    float *x = nullptr, *h = nullptr, *qv = nullptr, *kv = nullptr, *vv = nullptr, *q = nullptr, *att = nullptr, *o = nullptr,
          *x1 = nullptr, *mm = nullptr, *ee = nullptr, *rr = nullptr, *logits_r = nullptr, *wts = nullptr, *gate = nullptr,
          *upp = nullptr, *hd = nullptr, *mlp = nullptr, *gu = nullptr, *parts = nullptr, *parts_cpu = nullptr, *hfin = nullptr,
          *logits = nullptr, *sc = nullptr, *tab = nullptr, *pv = nullptr, *amax_v = nullptr;
    void *xq = nullptr, *hq = nullptr;
    uint8_t *ax = nullptr, *ah = nullptr;
    int32_t *ids = nullptr, *i32 = nullptr, *src = nullptr, *ents = nullptr, *pi = nullptr, *amax_i = nullptr;
    G::Group* groups = nullptr;
    float2 *cs_swa = nullptr, *cs_glb = nullptr;
    size_t ax_stride = 0, ah_stride = 0;
    // pinned host staging
    int32_t *h_i32 = nullptr, *h_ids = nullptr, *h_src = nullptr, *h_ents = nullptr, *h_amax = nullptr;
    uint8_t* h_ax = nullptr;
    float *h_parts = nullptr, *h_logits = nullptr;
    G::Group* h_groups = nullptr;
    cudaEvent_t ev_route = nullptr;
    std::unique_ptr<C::ExpertPool> pool;
    std::vector<CpuJob> jobs;
    // the row-split path's buffers (SplitCtx) and its switch: a layer with fewer missed experts than `split_below` (0: the
    // pool's thread count) has their rows split into about `split_tasks` tasks per thread
    float* sp_gu = nullptr;
    uint8_t* sp_ah = nullptr;
    int split_below = 0, split_tasks = 3;
    std::vector<int16_t> idx_of;   // per expert: its group (>= 0) or job (<= -2) of the layer being planned, -1: none
    Stats stats;
    // the cache's policy (see adapt())
    bool adapt_on = true;
    float promote_min = 2.0f, decay = 0.5f;
    int up_max = 4, decay_every = 64;
    int64_t next_decay = 0;
    bool profile = false;
    double prof[8] = {};

    void init(Model& model, int context, int threads) {
        m = &model;
        ctx = context;
        ring = m->n_swa + kTmax;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        const size_t T = kTmax, N = (size_t) m->n_embd, Kp = (size_t) m->top_k, V = (size_t) m->n_vocab;
        size_t max_qdim = 0, max_kdim = 0;
        for (Layer& y : m->layers) {
            max_qdim = std::max(max_qdim, (size_t) y.H * y.HD);
            max_kdim = std::max(max_kdim, (size_t) y.HKV * y.HD);
            y.cap = y.swa ? ring : ctx;
            y.kc = dalloc<__half>((size_t) y.HKV * y.cap * y.HD);
            y.vc = dalloc<__half>((size_t) y.HKV * y.cap * y.HD);
        }
        x = dalloc<float>(T * N);
        h = dalloc<float>(T * N);
        qv = dalloc<float>(T * max_qdim);
        kv = dalloc<float>(T * max_kdim);
        vv = dalloc<float>(T * max_kdim);
        q = dalloc<float>(T * max_qdim);
        att = dalloc<float>(T * max_qdim);
        o = dalloc<float>(T * N);
        x1 = dalloc<float>(T * N);
        mm = dalloc<float>(T * N);
        ee = dalloc<float>(T * N);
        rr = dalloc<float>(T * N);
        logits_r = dalloc<float>(T * (size_t) m->n_expert);
        wts = dalloc<float>(T * Kp);
        gate = dalloc<float>(T * (size_t) m->n_ff);
        upp = dalloc<float>(T * (size_t) m->n_ff);
        hd = dalloc<float>(T * (size_t) m->n_ff);
        mlp = dalloc<float>(T * N);
        gu = dalloc<float>(T * Kp * 2 * (size_t) m->n_ff_exp);
        parts = dalloc<float>(T * Kp * N);
        parts_cpu = dalloc<float>(T * Kp * N);
        hfin = dalloc<float>(T * N);
        logits = dalloc<float>(T * V);
        sc = dalloc<float>(T * 16 * (size_t) std::max(ring, ctx));
        tab = dalloc<float>(65536);
        CK(cudaMemcpy(tab, g4::ref::gelu_table(), 65536 * 4, cudaMemcpyHostToDevice));
        pv = dalloc<float>(T * G::kArgBlocks);
        pi = dalloc<int32_t>(T * G::kArgBlocks);
        amax_i = dalloc<int32_t>(T);
        amax_v = dalloc<float>(T);
        const size_t qn = std::max<size_t>(max_qdim, N);
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) qn, kTmax) + 64));
        CK(cudaMalloc(&hq, K::native_q8_1_bytes(m->n_ff, kTmax) + 64));
        ax_stride = g4::act_bytes(m->n_embd);
        ah_stride = g4::act_bytes(m->n_ff_exp);
        ax = dalloc<uint8_t>(ax_stride * T);
        ah = dalloc<uint8_t>(ah_stride * T * Kp);
        ids = dalloc<int32_t>(T * Kp);
        i32 = dalloc<int32_t>(T * 8);
        src = dalloc<int32_t>(T * Kp);
        ents = dalloc<int32_t>(T * Kp);
        groups = dalloc<G::Group>(T * Kp);
        cs_swa = dalloc<float2>(T * (size_t) m->hd_swa / 2);
        cs_glb = dalloc<float2>(T * (size_t) m->hd / 2);
        h_i32 = halloc<int32_t>(T * 8);
        h_ids = halloc<int32_t>(T * Kp);
        h_src = halloc<int32_t>(T * Kp);
        h_ents = halloc<int32_t>(T * Kp);
        h_amax = halloc<int32_t>(T);
        h_ax = halloc<uint8_t>(ax_stride * T);
        h_parts = halloc<float>(T * Kp * N);
        h_logits = halloc<float>(T * V);
        h_groups = halloc<G::Group>(T * Kp);
        jobs.resize(T * Kp);
        sp_gu = (float*) _aligned_malloc(T * Kp * kTmax * 2 * (size_t) m->n_ff_exp * sizeof(float), 64);
        sp_ah = (uint8_t*) _aligned_malloc(T * Kp * kTmax * ah_stride, 64);
        if (!sp_gu || !sp_ah) die("no host memory for the CPU experts' scratch");
        idx_of.assign((size_t) m->n_expert, -1);
        pool = std::make_unique<C::ExpertPool>(threads, true, true);
        std::fprintf(stderr, "strata-g4: context %d, sliding ring %d cells, CPU pool %d workers%s, AVX-VNNI %s\n", ctx, ring,
                     pool->workers(), pool->host_works() ? " + the host thread" : "", g4::cpu::have_avx_vnni() ? "yes" : "no");
    }

    /// The expert slots: `slots` per layer (-1: what fits in `vram_mib`, or in the free VRAM less a reserve), the first
    /// experts of every layer to start with.
    void place(int slots, double vram_mib) {
        size_t per_slot = 0;
        for (const Layer& y : m->layers) per_slot += y.f.stride;
        if (slots < 0) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t reserve = (size_t) 384 << 20;
            const size_t budget = vram_mib >= 0 ? (size_t) (vram_mib * 1048576.0) : (fr > reserve ? fr - reserve : 0);
            slots = (int) std::min<size_t>(budget / per_slot, (size_t) m->n_expert);
        }
        slots = std::max(0, std::min(slots, m->n_expert));
        const auto t0 = Clock::now();
        for (Layer& y : m->layers) {
            y.n_slots = slots;
            y.exp_in_slot.assign((size_t) slots, -1);
            if (slots == 0) continue;
            CK(cudaMalloc(&y.slots, y.f.stride * (size_t) slots));
            CK(cudaMemcpy(y.slots, y.hblobs, y.f.stride * (size_t) slots, cudaMemcpyHostToDevice));
            for (int s = 0; s < slots; ++s) {
                y.slot_of[(size_t) s] = (int16_t) s;
                y.exp_in_slot[(size_t) s] = (int16_t) s;
            }
        }
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-g4: expert cache: %d of %d experts per layer in VRAM (%.2f GiB), %.1f s; CUDA reports %.0f MiB "
                     "free of %.0f\n", slots, m->n_expert, (double) slots * per_slot / (1 << 30), ms_since(t0) / 1000.0,
                     fr / 1048576.0, tot / 1048576.0);
    }

    void psync(int k, Clock::time_point& t) {   // --profile: stage k ends here
        if (!profile) return;
        CK(cudaStreamSynchronize(st));
        prof[k] += ms_since(t);
        t = Clock::now();
    }

    void mmvq(const Mat& w, const void* in_q8, float* y, int T) { K::native_mmvq(w.type, w.d, in_q8, y, w.n_in, w.n_out, T, st); }

    /// The copies that have landed become residents; then, for the layer just planned, the experts in demand without
    /// a slot are copied over the least-used residents (after this layer's kernels of the current forward, which may
    /// still read the slots).  `routed`: the experts of this forward's rows in layer l; `missed`: those without a slot.
    void adapt(Layer& y, const std::vector<int>& missed) {
        if (!adapt_on || y.n_slots == 0 || y.up_busy || missed.empty()) return;
        std::vector<int> cand;
        for (int e : missed)
            if (y.score[(size_t) e] >= promote_min) cand.push_back(e);
        if (cand.empty()) return;
        std::sort(cand.begin(), cand.end(), [&](int a, int b) { return y.score[(size_t) a] > y.score[(size_t) b]; });
        int n = 0;
        for (int e : cand) {
            if (n >= up_max) break;
            int best = -1;
            float bs = y.score[(size_t) e];   // the victim must be used less than the candidate
            for (int s = 0; s < y.n_slots; ++s) {
                const int r = y.exp_in_slot[(size_t) s];
                bool taken = false;
                for (const auto& pr : y.pending) taken |= pr.second == s;
                if (taken) continue;
                const float v = r < 0 ? -1.0f : y.score[(size_t) r];
                if (v < bs) {
                    bs = v;
                    best = s;
                }
            }
            if (best < 0) break;
            const int old = y.exp_in_slot[(size_t) best];
            if (old >= 0) y.slot_of[(size_t) old] = -1;
            y.exp_in_slot[(size_t) best] = -1;
            y.pending.emplace_back(e, best);
            ++n;
        }
        if (n == 0) return;
        CK(cudaStreamWaitEvent(up, y.ev_exp, 0));
        for (const auto& pr : y.pending)
            CK(cudaMemcpyAsync(y.slots + (size_t) pr.second * y.f.stride, y.hblobs + (size_t) pr.first * y.f.stride, y.f.bytes,
                               cudaMemcpyHostToDevice, up));
        CK(cudaEventRecord(y.ev_up, up));
        y.up_busy = true;
        stats.uploads += n;
    }
    void land(Layer& y) {
        if (!y.up_busy || cudaEventQuery(y.ev_up) != cudaSuccess) return;
        for (const auto& pr : y.pending) {
            y.slot_of[(size_t) pr.first] = (int16_t) pr.second;
            y.exp_in_slot[(size_t) pr.second] = (int16_t) pr.first;
        }
        y.pending.clear();
        y.up_busy = false;
    }

    /// T rows (the tokens of cells pos .. pos + T - 1) through the model.  With `want_logits` every row's argmax lands
    /// in h_amax (and its logits stay in `logits`).  The caller advances `pos` by the rows it keeps.
    void forward(const int32_t* tokens, int T, bool want_logits) {
        const int N = m->n_embd, Kp = m->top_k, NE = m->n_expert, FF = m->n_ff_exp;
        if (T < 1 || T > kTmax) die("a forward takes 1.." + std::to_string(kTmax) + " rows");
        if (pos + T > ctx) die("the context (" + std::to_string(ctx) + " tokens) is full");
        auto tp = Clock::now();
        // tokens, positions and cache cells of the rows
        for (int t = 0; t < T; ++t) {
            const int p = (int) pos + t;
            const int cnt = std::min(p + 1, m->n_swa);
            h_i32[t] = tokens[t];
            h_i32[kTmax + t] = p;
            h_i32[2 * kTmax + t] = p % ring;                   // sliding cell
            h_i32[3 * kTmax + t] = (p - cnt + 1) % ring;       // first visible sliding cell
            h_i32[4 * kTmax + t] = cnt;
            h_i32[5 * kTmax + t] = p;                          // global cell
            h_i32[6 * kTmax + t] = 0;
            h_i32[7 * kTmax + t] = p + 1;
        }
        CK(cudaMemcpyAsync(i32, h_i32, (size_t) kTmax * 8 * 4, cudaMemcpyHostToDevice, st));
        const int32_t *d_tok = i32, *d_pos = i32 + kTmax, *d_cell_s = i32 + 2 * kTmax, *d_first_s = i32 + 3 * kTmax,
                      *d_cnt_s = i32 + 4 * kTmax, *d_cell_g = i32 + 5 * kTmax, *d_first_g = i32 + 6 * kTmax, *d_cnt_g = i32 + 7 * kTmax;
        G::k_embed<<<dim3((unsigned) (N / 32), (unsigned) T), 32, 0, st>>>(m->embd.d, d_tok, x, N, std::sqrt((float) N));
        G::k_rope_tab<<<(unsigned) T, (unsigned) (m->hd_swa / 2), 0, st>>>(d_pos, nullptr, cs_swa, m->hd_swa,
                                                                           std::pow(m->rope_base_swa, -2.0f / (float) m->hd_swa));
        G::k_rope_tab<<<(unsigned) T, (unsigned) (m->hd / 2), 0, st>>>(d_pos, m->rope_freqs, cs_glb, m->hd,
                                                                       std::pow(m->rope_base, -2.0f / (float) m->hd));
        const G::ActDev dax{ax, ax_stride, N}, dah{ah, ah_stride, FF};
        std::vector<int> missed;
        for (int l = 0; l < m->n_layer; ++l) {
            Layer& y = m->layers[(size_t) l];
            const int QD = y.H * y.HD, KD = y.HKV * y.HD;
            // ---- attention
            G::k_rms_w<<<(unsigned) T, G::kThreads, 0, st>>>(x, y.attn_norm, h, N, m->eps);
            K::native_quantize_q8_1(h, xq, N, T, st);
            mmvq(y.wq, xq, qv, T);
            mmvq(y.wk, xq, kv, T);
            if (y.has_v) mmvq(y.wv, xq, vv, T);
            G::k_attn_prep<<<dim3((unsigned) (y.H + 2 * y.HKV), (unsigned) T), G::kThreads, 0, st>>>(
                qv, kv, y.has_v ? vv : kv, y.q_norm, y.k_norm, y.swa ? cs_swa : cs_glb, q, y.kc, y.vc,
                y.swa ? d_cell_s : d_cell_g, y.H, y.HKV, y.HD, y.cap, m->eps);
            G::k_attn<<<dim3((unsigned) y.H, (unsigned) T), G::kThreads, 0, st>>>(
                q, y.kc, y.vc, y.swa ? d_first_s : d_first_g, y.swa ? d_cnt_s : d_cnt_g, sc, att, y.H, y.HKV, y.HD, y.cap);
            K::native_quantize_q8_1(att, xq, QD, T, st);
            mmvq(y.wo, xq, o, T);
            (void) KD;
            psync(0, tp);
            // ---- the norms of x1, the router, the experts' input; the routing and that input to the host
            G::k_post_attn<<<(unsigned) T, G::kThreads, 0, st>>>(x, o, y.post_attn_norm, y.ffn_norm, y.pre_ffw_norm_2, y.router_scale,
                                                                x1, mm, ee, rr, N, m->eps, 1.0f / std::sqrt((float) N));
            G::k_router_logits<<<dim3((unsigned) ((NE + 7) / 8), (unsigned) T), G::kThreads, 0, st>>>(y.router, rr, logits_r, N, NE);
            G::k_router_topk<<<(unsigned) T, G::kThreads, 0, st>>>(logits_r, ids, wts, NE, Kp);
            G::k_quant_a8<<<dim3((unsigned) (N / 32), (unsigned) T), 32, 0, st>>>(ee, N, dax);
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * Kp * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_ax, ax, ax_stride * (size_t) T, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
            // ---- the dense FFN (the GPU works on it while the host plans the experts)
            K::native_quantize_q8_1(mm, xq, N, T, st);
            if (!K::native_mmvq_pair(y.ffn_gate.type, y.ffn_gate.d, y.ffn_up.d, xq, gate, upp, N, m->n_ff, T, st)) {
                mmvq(y.ffn_gate, xq, gate, T);
                mmvq(y.ffn_up, xq, upp, T);
            }
            G::k_geglu_dense<<<(unsigned) (((int64_t) T * m->n_ff + G::kThreads - 1) / G::kThreads), G::kThreads, 0, st>>>(
                gate, upp, tab, hd, (int64_t) T * m->n_ff);
            K::native_quantize_q8_1(hd, hq, m->n_ff, T, st);
            mmvq(y.ffn_down, hq, mlp, T);
            // ---- the host: this layer's picks, split into the slots' groups (GPU) and the others (CPU pool)
            const auto tw = Clock::now();
            while (cudaEventQuery(ev_route) == cudaErrorNotReady) _mm_pause();
            const auto tq = Clock::now();
            stats.ms_wait_route += std::chrono::duration<double, std::milli>(tq - tw).count();
            land(y);
            int n_groups = 0, n_jobs = 0, n_res = 0, n_miss = 0;
            missed.clear();
            for (int e = 0; e < T * Kp; ++e) {
                const int xp = h_ids[e], t = e / Kp;
                y.score[(size_t) xp] += 1.0f;
                int16_t& ix = idx_of[(size_t) xp];
                if (y.slot_of[(size_t) xp] >= 0) {
                    if (ix == -1) {
                        ix = (int16_t) n_groups;
                        G::Group& g = h_groups[n_groups++];
                        g.blob = y.slots + (size_t) y.slot_of[(size_t) xp] * y.f.stride;
                        g.nt = 0;
                    }
                    G::Group& g = h_groups[ix];
                    g.ent[g.nt] = e;
                    g.row[g.nt] = t;
                    ++g.nt;
                    h_ents[n_res++] = e;
                    h_src[e] = -1;
                } else {
                    if (ix == -1) {
                        ix = (int16_t) (-2 - n_jobs);
                        CpuJob& j = jobs[(size_t) n_jobs++];
                        j.blob = y.hblobs + (size_t) xp * y.f.stride;
                        j.nt = 0;
                        missed.push_back(xp);
                    }
                    CpuJob& j = jobs[(size_t) (-2 - ix)];
                    j.act[j.nt] = h_ax + (size_t) t * ax_stride;
                    j.out[j.nt] = h_parts + (size_t) n_miss * N;
                    ++j.nt;
                    h_src[e] = n_miss++;
                }
            }
            for (int e = 0; e < T * Kp; ++e) idx_of[(size_t) h_ids[e]] = -1;
            stats.ent_gpu += n_res;
            stats.ent_cpu += n_miss;
            stats.cpu_jobs += n_jobs;
            CK(cudaMemcpyAsync(src, h_src, (size_t) T * Kp * 4, cudaMemcpyHostToDevice, st));
            if (n_groups > 0) {
                CK(cudaMemcpyAsync(groups, h_groups, (size_t) n_groups * sizeof(G::Group), cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(ents, h_ents, (size_t) n_res * 4, cudaMemcpyHostToDevice, st));
                const dim3 gg((unsigned) (2 * FF / 32), (unsigned) n_groups), gd((unsigned) (N / 32), (unsigned) n_groups);
                if (y.f.gu_type == g4::kTypeQ5_K) G::k_gate_up<true><<<gg, 128, 0, st>>>(groups, dax, gu, N, FF);
                else G::k_gate_up<false><<<gg, 128, 0, st>>>(groups, dax, gu, N, FF);
                G::k_geglu_quant<<<dim3((unsigned) (FF / 32), (unsigned) n_res), 32, 0, st>>>(ents, gu, tab, dah, FF);
                if (y.f.d_type == g4::kTypeQ8_0) G::k_down<true><<<gd, 256, 0, st>>>(groups, dah, parts, y.f.down_off, N, FF);
                else G::k_down<false><<<gd, 256, 0, st>>>(groups, dah, parts, y.f.down_off, N, FF);
            }
            CK(cudaEventRecord(y.ev_exp, st));
            stats.ms_plan += ms_since(tq);
            if (n_jobs > 0) {
                const auto tc = Clock::now();
                const int threads = pool->workers() + (pool->host_works() ? 1 : 0);
                if (n_jobs < (split_below > 0 ? split_below : threads)) {
                    SplitCtx sx;
                    sx.f = &y.f;
                    sx.jobs = jobs.data();
                    sx.chunks = std::max(1, std::min(64, (threads * split_tasks + n_jobs - 1) / n_jobs));
                    sx.gu = sp_gu;
                    sx.ah = sp_ah;
                    sx.ah_stride = ah_stride;
                    pool->run_range(n_jobs * sx.chunks, split_gu_fn, &sx);
                    if (n_jobs > 2) pool->run_range(n_jobs, split_mid_fn, &sx);
                    else
                        for (int j = 0; j < n_jobs; ++j) split_mid_fn(&sx, j);
                    pool->run_range(n_jobs * sx.chunks, split_down_fn, &sx);
                } else {
                    CpuCtx cc{&y.f, jobs.data()};
                    pool->run_range(n_jobs, cpu_job_fn, &cc);
                }
                stats.ms_cpu += ms_since(tc);
                CK(cudaMemcpyAsync(parts_cpu, h_parts, (size_t) n_miss * N * 4, cudaMemcpyHostToDevice, st));
            }
            adapt(y, missed);
            // ---- the layer's tail
            G::k_ffn_out<<<(unsigned) T, G::kThreads, 0, st>>>(x1, mlp, parts, parts_cpu, src, ids, wts, y.exp_scale, Kp,
                                                              y.post_ffw_norm_1, y.post_ffw_norm_2, y.post_ffw_norm, y.out_scale, x,
                                                              N, m->eps);
            psync(1, tp);
        }
        if (want_logits) {
            G::k_rms_w<<<(unsigned) T, G::kThreads, 0, st>>>(x, m->out_norm, hfin, N, m->eps);
            K::native_quantize_q8_1(hfin, xq, N, T, st);
            mmvq(m->embd, xq, logits, T);
            G::k_argmax1<<<dim3(G::kArgBlocks, (unsigned) T), G::kThreads, 0, st>>>(logits, m->n_vocab, pv, pi);
            G::k_argmax2<<<(unsigned) T, G::kArgBlocks, 0, st>>>(pv, pi, amax_i, amax_v);
            CK(cudaMemcpyAsync(h_amax, amax_i, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        if (profile) prof[2] += ms_since(tp);
        ++stats.passes;
        stats.rows += T;
        // the counts age: every decay_every positions
        if (pos + T >= next_decay) {
            for (Layer& y : m->layers)
                for (float& v : y.score) v *= decay;
            next_decay = pos + T + decay_every;
        }
    }

    /// Row r's logits of the last forward, soft-capped as the model defines them, into `out` (n_vocab floats).
    void read_logits(int r, float* out) {
        CK(cudaMemcpy(out, logits + (size_t) r * m->n_vocab, (size_t) m->n_vocab * 4, cudaMemcpyDeviceToHost));
        if (m->softcap > 0.0f)
            for (int i = 0; i < m->n_vocab; ++i) out[i] = std::tanh(out[i] / m->softcap) * m->softcap;
    }
};

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> v;
    const char* p = s.c_str();
    while (*p) {
        while (*p && !(*p == '-' || (*p >= '0' && *p <= '9'))) ++p;
        if (!*p) break;
        v.push_back((int32_t) std::strtol(p, (char**) &p, 10));
    }
    return v;
}
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
/// A file of ids: either plain numbers, or g4-oracle's stdout (its "prompt:" line, or with `key` "gen", its "gen:" line).
std::vector<int32_t> read_ids(const std::string& path, const char* key) {
    const std::string all = read_file(path);
    const std::string tag = std::string(key) + ":";
    const size_t at = all.find(tag);
    if (at == std::string::npos) return parse_ids(all);
    const size_t end = all.find('\n', at);
    return parse_ids(all.substr(at + tag.size(), end == std::string::npos ? std::string::npos : end - at - tag.size()));
}

}  // namespace

int strata::gemma4::run(int argc, char** argv) {
    std::string model, tokens_file, force_file, top2_file, full_file, bench_json, jobs_file;
    int n_gen = 32, ctx = 4096, threads = 0, slots = -1, dev = 0, reps = 1;
    double vram_mib = -1;
    Engine E;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = val();
        else if (a == "--tokens-file") tokens_file = val();
        else if (a == "--force-file") force_file = val();
        else if (a == "--top2") top2_file = val();
        else if (a == "--full-logits") full_file = val();
        else if (a == "--bench-json") bench_json = val();
        else if (a == "--jobs-file") jobs_file = val();
        else if (a == "--reps") reps = std::atoi(val().c_str());
        else if (a == "--n") n_gen = std::atoi(val().c_str());
        else if (a == "--ctx") ctx = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--slots") slots = std::atoi(val().c_str());
        else if (a == "--expert-vram-mib") vram_mib = std::atof(val().c_str());
        else if (a == "--device") dev = std::atoi(val().c_str());
        else if (a == "--adapt") E.adapt_on = std::atoi(val().c_str()) != 0;
        else if (a == "--promote-min") E.promote_min = (float) std::atof(val().c_str());
        else if (a == "--up-max") E.up_max = std::atoi(val().c_str());
        else if (a == "--decay") E.decay = (float) std::atof(val().c_str());
        else if (a == "--decay-every") E.decay_every = std::atoi(val().c_str());
        else if (a == "--profile") E.profile = std::atoi(val().c_str()) != 0;
        else if (a == "--split-below") E.split_below = std::atoi(val().c_str());
        else if (a == "--split-tasks") E.split_tasks = std::atoi(val().c_str());
        else if (a == "--help" || a == "-h") {
            std::printf("strata-g4 --model GGUF --tokens-file F [--n N] [--ctx 4096] [--threads N] [--slots S | --expert-vram-mib M]\n"
                        "          [--adapt 0|1] [--promote-min X] [--up-max N] [--decay X] [--decay-every N] [--profile 0|1]\n"
                        "          [--force-file F] [--top2 F] [--full-logits F] [--bench-json F] [--device N]\n"
                        "          [--jobs-file F [--reps R]]   lines: <id> <tokens file> <new tokens> [<forced ids file>]\n");
            return 0;
        } else die("unknown option " + a + " (see --help)");
    }
    if (model.empty() || (tokens_file.empty() && jobs_file.empty())) die("--model and --tokens-file (or --jobs-file) are required (see --help)");
    if (!g4::cpu::available()) die("this CPU has no AVX2 + FMA + F16C");
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    // the requests: one (--tokens-file), or the lines of --jobs-file: <id> <tokens file> <new tokens> [<forced ids file>]
    struct Job {
        std::string id;
        std::vector<int32_t> prompt, forced;
        int n_gen = 0;
    };
    std::vector<Job> jobs;
    auto add_job = [&](const std::string& id, const std::string& tf, int n, const std::string& ff) {
        Job j;
        j.id = id;
        j.prompt = read_ids(tf, "prompt");
        if (j.prompt.empty()) die("no token ids in " + tf);
        if (!ff.empty()) j.forced = read_ids(ff, "gen");
        j.n_gen = j.forced.empty() ? n : (int) j.forced.size() + 1;
        if ((int) j.prompt.size() + j.n_gen > ctx) die(id + ": the prompt and the new tokens do not fit the context");
        jobs.push_back(std::move(j));
    };
    if (!jobs_file.empty()) {
        std::istringstream in(read_file(jobs_file));
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string id, tf, ff;
            int n = 0;
            if (!(ls >> id >> tf >> n) || id[0] == '#') continue;
            ls >> ff;
            add_job(id, tf, n, ff);
        }
        if (jobs.empty()) die("no jobs in " + jobs_file);
    } else add_job("single", tokens_file, n_gen, force_file);
    const bool single = jobs_file.empty();

    Model M;
    load(model, M);
    E.init(M, ctx, threads);
    E.place(slots, vram_mib);
    const C::ThreadAffinity aff = C::pin_current_thread(C::planned_host_core());
    (void) aff;

    FILE* f_top2 = top2_file.empty() ? nullptr : std::fopen(top2_file.c_str(), "wb");
    FILE* f_full = full_file.empty() ? nullptr : std::fopen(full_file.c_str(), "wb");
    std::vector<float> lg(f_top2 || f_full ? (size_t) M.n_vocab : 0);
    auto dump = [&](int row) {
        if (!f_top2 && !f_full) return;
        E.read_logits(row, lg.data());
        if (f_full) std::fwrite(lg.data(), 4, lg.size(), f_full);
        if (f_top2) {
            int a = 0, c = -1;
            for (int j = 1; j < M.n_vocab; ++j)
                if (lg[(size_t) j] > lg[(size_t) a]) a = j;
            for (int j = 0; j < M.n_vocab; ++j)
                if (j != a && (c < 0 || lg[(size_t) j] > lg[(size_t) c])) c = j;
            const float rec[4] = {(float) a, lg[(size_t) a], (float) c, lg[(size_t) c]};
            std::fwrite(rec, 4, 4, f_top2);
        }
    };
    FILE* f_json = bench_json.empty() ? nullptr : std::fopen(bench_json.c_str(), "wb");
    if (f_json)
        std::fprintf(f_json, "{\"slots\": %d, \"threads\": %d, \"host_works\": %s, \"context\": %d, \"runs\": [\n", M.layers[0].n_slots,
                     E.pool->workers(), E.pool->host_works() ? "true" : "false", ctx);
    bool first_run = true;
    for (int rep = 0; rep < reps; ++rep)
        for (const Job& job : jobs) {
            const std::vector<int32_t>& prompt = job.prompt;
            const int ng = job.n_gen;
            E.pos = 0;   // a new conversation; the expert cache keeps what it learned
            E.next_decay = 0;
            E.stats.reset();
            if (single) {
                std::printf("prompt:");
                for (int32_t t : prompt) std::printf(" %d", t);
                std::printf("\n");
            }
            // the prompt, kTmax rows at a time
            const auto t_prompt = Clock::now();
            int32_t next = 0;
            for (size_t at = 0; at < prompt.size(); at += kTmax) {
                const int T = (int) std::min<size_t>(kTmax, prompt.size() - at);
                const bool last = at + T >= prompt.size();
                E.forward(prompt.data() + at, T, last);
                E.pos += T;
                if (last) {
                    next = E.h_amax[T - 1];
                    dump(T - 1);
                }
            }
            const double prompt_ms = ms_since(t_prompt);
            const Stats prompt_stats = E.stats;
            E.stats.reset();
            for (double& v : E.prof) v = 0;

            std::vector<int32_t> gen;
            gen.push_back(next);
            const auto t_gen = Clock::now();
            for (int i = 0; i + 1 < ng; ++i) {
                const int32_t tok = job.forced.empty() ? next : job.forced[(size_t) i];
                E.forward(&tok, 1, true);
                E.pos += 1;
                next = E.h_amax[0];
                gen.push_back(next);
                dump(0);
            }
            const double gen_ms = ms_since(t_gen);
            if (single) {
                std::printf("gen:");
                for (int32_t t : gen) std::printf(" %d", t);
                std::printf("\n");
                std::fflush(stdout);
            }
            const Stats& s = E.stats;
            const double tok_s = ng > 1 ? (ng - 1) * 1000.0 / gen_ms : 0.0;
            const double hit = s.ent_gpu + s.ent_cpu ? (double) s.ent_gpu / (double) (s.ent_gpu + s.ent_cpu) : 0.0;
            const double pp = s.passes ? (double) s.passes : 1.0;
            std::fprintf(stderr, "strata-g4: [%s rep %d] prompt %zu tokens in %.1f ms (%.1f tok/s); decode %d tokens in %.1f ms = "
                         "%.2f tok/s (%.2f ms per pass)\n", job.id.c_str(), rep, prompt.size(), prompt_ms,
                         prompt.size() * 1000.0 / prompt_ms, ng - 1, gen_ms, tok_s, gen_ms / pp);
            std::fprintf(stderr, "strata-g4:   decode experts: %.1f%% on the GPU (%lld of %lld picks), CPU pool %.2f ms per pass in "
                         "%.1f jobs, wait for routing %.2f ms, plan %.2f ms per pass; %lld experts copied to VRAM (prompt: %.1f%% on "
                         "the GPU, %lld copied)\n", 100.0 * hit, s.ent_gpu, s.ent_gpu + s.ent_cpu, s.ms_cpu / pp, (double) s.cpu_jobs / pp,
                         s.ms_wait_route / pp, s.ms_plan / pp, s.uploads,
                         prompt_stats.ent_gpu + prompt_stats.ent_cpu
                             ? 100.0 * prompt_stats.ent_gpu / (double) (prompt_stats.ent_gpu + prompt_stats.ent_cpu) : 0.0,
                         prompt_stats.uploads);
            if (E.profile && s.passes)
                std::fprintf(stderr, "strata-g4:   profile per pass: attention %.2f ms, FFN block (router, dense, experts, tail) "
                             "%.2f ms, head %.2f ms\n", E.prof[0] / pp, E.prof[1] / pp, E.prof[2] / pp);
            if (f_json) {
                std::fprintf(f_json, "%s{\"id\": \"%s\", \"rep\": %d, \"prompt_tokens\": %zu, \"prompt_ms\": %.3f, \"decode_tokens\": %d, "
                             "\"decode_ms\": %.3f, \"decode_tok_s\": %.4f, \"passes\": %lld, \"gpu_pick_share\": %.5f, "
                             "\"cpu_ms_per_pass\": %.4f, \"wait_route_ms_per_pass\": %.4f, \"plan_ms_per_pass\": %.4f, \"uploads\": %lld, "
                             "\"prompt_uploads\": %lld, \"gen\": [", first_run ? "" : ",\n", job.id.c_str(), rep, prompt.size(), prompt_ms,
                             ng - 1, gen_ms, tok_s, s.passes, hit, s.ms_cpu / pp, s.ms_wait_route / pp, s.ms_plan / pp, s.uploads,
                             prompt_stats.uploads);
                for (size_t i = 0; i < gen.size(); ++i) std::fprintf(f_json, "%s%d", i ? ", " : "", gen[i]);
                std::fprintf(f_json, "]}");
                std::fflush(f_json);
                first_run = false;
            }
        }
    if (f_json) {
        std::fprintf(f_json, "\n]}\n");
        std::fclose(f_json);
    }
    if (f_top2) std::fclose(f_top2);
    if (f_full) std::fclose(f_full);
    return 0;
}
