#include "models.hpp"
#ifdef HV15N_ROCM
#include "../../rdna4/video_common/timestep_basis.hpp"
#endif
namespace hv15n {
#ifdef HV15N_ROCM
// Encoder and scheduler buffers remain FP32; only the model evaluation follows
// upstream's FP16 autocast. Restore the scope on exceptions and repeated calls.
struct DitPrecision {
    Gpu &gpu;
    bool previous;
    explicit DitPrecision(Gpu &g) : gpu(g), previous(g.dit_fp16) { g.dit_fp16 = true; }
    ~DitPrecision() { gpu.dit_fp16 = previous; }
};
#define HV15_DIT_PRECISION(g) DitPrecision precision_scope(g)
#else
#define HV15_DIT_PRECISION(g) ((void)0)
#endif
static Tensor time_embed(Gpu &g, Weights &w, const std::string &prefix,
                         float time) {
#ifdef HV15N_ROCM
  auto basis = g.upload(video_rocm::hv15_time_basis(), {1, 128});
  auto embedding = g.empty({1, 256});
  g.launch("hv15_time_embedding", 1, 1, 1, 128, 1, 0, embedding.pointer, basis.pointer, time);
  return g.linear(w, prefix + ".mlp.2",
                  g.op(g.linear(w, prefix + ".mlp.0", embedding), 3));
#else
  std::vector<float> values(256);
  for (int i = 0; i < 128; i++) {
    float a = time * std::exp(-std::log(10000.f) * i / 128);
    values[i] = std::cos(a);
    values[i + 128] = std::sin(a);
  }
  return g.linear(
      w, prefix + ".mlp.2",
      g.op(g.linear(w, prefix + ".mlp.0", g.upload(values, {1, 256})), 3));
#endif
}
static Tensor mlp(Gpu &g, Weights &w, const std::string &prefix,
                  const Tensor &x, int activation) {
  auto hidden = g.linear(w, prefix + ".fc1", x);
  hidden = g.optimized ? g.activate(std::move(hidden), activation) : g.op(hidden, activation);
  return g.linear(w, prefix + ".fc2", hidden);
}
static Tensor head_norm(Gpu &g, Weights &w, const std::string &prefix, Tensor x,
                        int heads) {
  auto shape = x.shape;
  int channels = x.channels();
  x.shape = {x.rows() * heads, channels / heads};
  auto y = g.norm(w, prefix, x, 1);
  y.shape = shape;
  return y;
}
static Tensor modulate(Gpu &g, const Tensor &x, const Tensor &mod, int start) {
  if (g.optimized) return g.modulate(x,mod,start);
  auto shift = g.columns(mod, start, x.channels()),
       scale = g.columns(mod, start + x.channels(), x.channels());
  return g.op(g.bare_norm(x), 7, &scale, &shift);
}
static Tensor gate_residual(Gpu &g, const Tensor &x, const Tensor &delta,
                            const Tensor &mod, int start) {
  if (g.optimized) return g.gated(x,delta,mod,start);
  auto gate = g.columns(mod, start, x.channels());
  auto change = g.op(delta, 8, &gate);
  return g.op(x, 1, &change);
}
static Tensor refined_text(Gpu &g, Weights &w, const Tensor &text, float time) {
  auto pool = g.mean(text);
  auto c = g.linear(w, "txt_in.c_embedder.linear_2",
                    g.op(g.linear(w, "txt_in.c_embedder.linear_1", pool), 3));
  auto t = time_embed(g, w, "txt_in.t_embedder", time);
  c = g.op(c, 1, &t);
  auto x = g.linear(w, "txt_in.input_embedder", text);
  for (int i = 0; i < 2; i++) {
    auto p = "txt_in.individual_token_refiner.blocks." + std::to_string(i);
    auto mod = g.linear(w, p + ".adaLN_modulation.1", g.op(c, 3));
    auto qkv = g.linear(w, p + ".self_attn_qkv", g.norm(w, p + ".norm1", x));
    auto q = g.columns(qkv, 0, 2048), k = g.columns(qkv, 2048, 2048),
         v = g.columns(qkv, 4096, 2048);
    auto delta =
        g.linear(w, p + ".self_attn_proj", g.attention(q, k, v, 16, 16));
    x = gate_residual(g, x, delta, mod, 0);
    delta = mlp(g, w, p + ".mlp", g.norm(w, p + ".norm2", x), 3);
    x = gate_residual(g, x, delta, mod, 2048);
  }
  return x;
}
std::pair<Tensor, Tensor> dit_block(Gpu &g, Weights &w, int index, Tensor img, Tensor txt,
                                    const Tensor &vec, int height, int width) {
  HV15_DIT_PRECISION(g);
  require(index >= 0 && index < 54, "DiT block index");
    g.poll();
    auto p = "double_blocks." + std::to_string(index);
    auto active = g.op(vec, 3);
    auto im = g.linear(w, p + ".img_mod.linear", active),
         tm = g.linear(w, p + ".txt_mod.linear", active);
    auto iqkv = g.linear(w, p + ".img_attn_qkv", modulate(g, img, im, 0)),
         tqkv = g.linear(w, p + ".txt_attn_qkv", modulate(g, txt, tm, 0));
    Tensor q,k,v;
    if (g.optimized && !g.vendor) {
      auto image = g.qkv_heads(w,p+".img_attn",iqkv,height,width,true);
      auto text = g.qkv_heads(w,p+".txt_attn",tqkv,height,width,false);
      iqkv={}; tqkv={};
      q=g.concat(image[0],text[0]);k=g.concat(image[1],text[1]);v=g.concat(image[2],text[2]);
    } else {
    auto iq = head_norm(g, w, p + ".img_attn_q_norm", g.columns(iqkv, 0, 2048),
                        16),
         ik = head_norm(g, w, p + ".img_attn_k_norm",
                        g.columns(iqkv, 2048, 2048), 16),
         iv = g.columns(iqkv, 4096, 2048);
    auto tq = head_norm(g, w, p + ".txt_attn_q_norm", g.columns(tqkv, 0, 2048),
                        16),
         tk = head_norm(g, w, p + ".txt_attn_k_norm",
                        g.columns(tqkv, 2048, 2048), 16),
         tv = g.columns(tqkv, 4096, 2048);
    iqkv = {};
    tqkv = {};
    g.rotary(iq, 16, 1, height, width, 256.f);
    g.rotary(ik, 16, 1, height, width, 256.f);
    q = g.concat(iq, tq); k = g.concat(ik, tk); v = g.concat(iv, tv);
    iq = {};
    ik = {};
    iv = {};
    tq = {};
    tk = {};
    tv = {};
    }
    auto attention = g.attention(q, k, v, 16, 16);
    q = {};
    k = {};
    v = {};
    auto delta =
        g.linear(w, p + ".img_attn_proj", g.rows(attention, 0, img.rows()));
    auto text_delta = g.linear(w, p + ".txt_attn_proj",
                               g.rows(attention, img.rows(), txt.rows()));
    attention = {};
    img = gate_residual(g, img, delta, im, 4096);
    delta = mlp(g, w, p + ".img_mlp", modulate(g, img, im, 6144), 4);
    img = gate_residual(g, img, delta, im, 10240);
    txt = gate_residual(g, txt, text_delta, tm, 4096);
    delta = mlp(g, w, p + ".txt_mlp", modulate(g, txt, tm, 6144), 4);
    txt = gate_residual(g, txt, delta, tm, 10240);
  return {img, txt};
}
void dit_blocks(Gpu &g, Weights &w, std::vector<DitState> &states,
                int first, int count, int height, int width) {
    require(first >= 0 && count > 0 && first + count <= 54 && !states.empty(),
            "DiT block range/states");
    g.prefetch_block(w, first);
    for (int index = first; index < first + count; ++index) {
        if (index + 1 < first + count) g.prefetch_block(w, index + 1);
        g.wait_block(index);
        for (auto &state : states) {
            auto output = dit_block(g, w, index, state.img, state.txt, state.vec, height, width);
            state.img = std::move(output.first);
            state.txt = std::move(output.second);
        }
        if (index > first) g.release_block(w, index - 1);
    }
    g.release_block(w, first + count - 1);
}
static DitState prepare_dit(Gpu &g, Weights &w, const Tensor &latent, const Tensor &conditioning,
           const Tensor &text, const Tensor &glyph, const Tensor &vision,
           float time, float next_time) {
  HV15_DIT_PRECISION(g);
  require(latent.shape.size() == 4 && latent.channels() == 32 &&
              conditioning.channels() == 33 &&
              latent.rows() == conditioning.rows(),
          "DiT conditioning shape");
  auto input = g.empty({latent.shape[0], latent.shape[1], latent.shape[2], 65});
  g.launch("join_condition", int((input.count() + 255) / 256), 1, 1, 256, 1, 0,
           input.pointer, latent.pointer, conditioning.pointer, latent.rows());
  auto img = g.conv(w, "img_in.proj", input, false);
  img.shape = {img.rows(), 2048};
  auto vec = time_embed(g, w, "time_in", time);
  if (w.has("time_r_in.mlp.0.weight")) {
    auto t = time_embed(g, w, "time_r_in", next_time);
    vec = g.op(vec, 1, &t);
  }
  auto txt = refined_text(g, w, text, time);
  auto types = g.weight(w, "cond_type_embedding.weight");
  auto type = g.rows(types, 0, 1);
  txt = g.op(txt, 6, nullptr, &type);
  if (glyph.pointer) {
    auto z = g.norm(w, "byt5_in.layernorm", glyph, 0, 1.e-5f);
    z = g.linear(w, "byt5_in.fc1", z);
    z = g.op(z, 5);
    z = g.linear(w, "byt5_in.fc2", z);
    z = g.op(z, 5);
    z = g.linear(w, "byt5_in.fc3", z);
    type = g.rows(types, 1, 1);
    z = g.op(z, 6, nullptr, &type);
    txt = g.concat(z, txt);
  }
  if (vision.pointer) {
    auto z = g.norm(w, "vision_in.proj.0", vision, 0, 1.e-5f);
    z = g.op(g.linear(w, "vision_in.proj.1", z), 5);
    z = g.linear(w, "vision_in.proj.3", z);
    z = g.norm(w, "vision_in.proj.4", z, 0, 1.e-5f);
    type = g.rows(types, 2, 1);
    z = g.op(z, 6, nullptr, &type);
    txt = g.concat(z, txt);
  }
  return {img,txt,vec,latent.shape};
}
Tensor dit_finish(Gpu &g,Weights &w,const DitState &state) {
  HV15_DIT_PRECISION(g);
  auto mod = g.linear(w, "final_layer.adaLN_modulation.1", g.op(state.vec, 3));
  auto out = g.linear(w, "final_layer.linear", modulate(g, state.img, mod, 0));
  out.shape = state.shape;
  return out;
}
Tensor dit(Gpu &g, Weights &w, const Tensor &latent, const Tensor &conditioning,
           const Tensor &text, const Tensor &glyph, const Tensor &vision, float time, float next_time) {
  std::vector<DitState> states{prepare_dit(g,w,latent,conditioning,text,glyph,vision,time,next_time)};
  dit_blocks(g,w,states,0,54,latent.shape[1],latent.shape[2]);
  return dit_finish(g,w,states[0]);
}
std::pair<Tensor,Tensor> dit_pair(Gpu &g,Weights &w,const Tensor &latent,const Tensor &conditioning,
                                 const Tensor &text,const Tensor &negative,const Tensor &glyph,
                                 const Tensor &vision,float time,float next_time) {
  if(!g.optimized || g.vendor) {
    Tensor empty;
    return {dit(g,w,latent,conditioning,text,glyph,vision,time,next_time),
            dit(g,w,latent,conditioning,negative,empty,vision,time,next_time)};
  }
  Tensor empty;
  std::vector<DitState> states{
      prepare_dit(g,w,latent,conditioning,text,glyph,vision,time,next_time),
      prepare_dit(g,w,latent,conditioning,negative,empty,vision,time,next_time)};
  dit_blocks(g,w,states,0,54,latent.shape[1],latent.shape[2]);
  return {dit_finish(g,w,states[0]),dit_finish(g,w,states[1])};
}
} // namespace hv15n
