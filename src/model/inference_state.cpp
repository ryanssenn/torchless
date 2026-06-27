#include "model/inference_state.h"

#include <cmath>

namespace {

void init_inv_freq(Tensor& inv_freq, float rope_theta) {
    for (size_t i = 0; i < inv_freq.numel; i++) {
        inv_freq.f32()[i] = 1.0f / std::pow(rope_theta, float(i) / inv_freq.numel);
    }
}

} // namespace

InferenceState::InferenceState(const Config& config)
    : config(config),
      arena(256 * 1024 * 1024),
      hidden_state(Tensor::alloc(arena, DType::F32, {config.hidden_size})),
      residual(Tensor::alloc(arena, DType::F32, {config.hidden_size})),

      inv_freq(Tensor::alloc(arena, DType::F32, {config.head_dim / 2})),
      cos(Tensor::alloc(arena, DType::F32, {config.head_dim})),
      sin(Tensor::alloc(arena, DType::F32, {config.head_dim})),

      q_state(Tensor::alloc(arena, DType::F32, {config.n_heads, config.head_dim})),
      k_state(Tensor::alloc(arena, DType::F32, {config.n_kv_heads, config.head_dim})),
      v_state(Tensor::alloc(arena, DType::F32, {config.n_kv_heads, config.head_dim})),

      k_cache(Tensor::alloc(arena, DType::F16, {config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, config.head_dim})),
      v_cache(Tensor::alloc(arena, DType::F16, {config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, config.head_dim})),

      scores(Tensor::alloc(arena, DType::F32, {config.n_heads, MAX_SEQ_LEN})),
      context(Tensor::alloc(arena, DType::F32, {config.n_heads, config.head_dim})),

      mlp_gate(Tensor::alloc(arena, DType::F32, {config.intermediate_size})),
      mlp_up(Tensor::alloc(arena, DType::F32, {config.intermediate_size})),

      logits(Tensor::alloc(arena, DType::F32, {config.vocab_size})),
      probs(Tensor::alloc(arena, DType::F32, {config.vocab_size})) {
    init_inv_freq(inv_freq, config.rope_theta);
}

void InferenceState::push_kv(size_t layer) {
    for (size_t h = 0; h < config.n_kv_heads; h++) {
        Tensor k_dst = k_cache.at({layer, h, pos});
        Tensor v_dst = v_cache.at({layer, h, pos});
        Tensor k_src = k_state.at({h});
        Tensor v_src = v_state.at({h});

        for (size_t j = 0; j < config.head_dim; j++) {
            k_dst.f16()[j] = static_cast<__fp16>(k_src.f32()[j]);
            v_dst.f16()[j] = static_cast<__fp16>(v_src.f32()[j]);
        }
    }
}
