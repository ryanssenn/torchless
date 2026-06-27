#pragma once

#include "common/arena.h"
#include "common/tensor.h"
#include "loader/model_load.h"

inline constexpr size_t MAX_SEQ_LEN = 500;

struct InferenceState {
    Config config;
    Arena arena;

    Tensor hidden_state; // [hidden_size]
    Tensor residual; // [hidden_size]
    size_t pos = 0;

    Tensor inv_freq; // [head_dim / 2]
    Tensor cos; // [head_dim]
    Tensor sin; // [head_dim]

    Tensor q_state; // [n_heads, head_dim]
    Tensor k_state; // [n_kv_heads, head_dim]
    Tensor v_state; // [n_kv_heads, head_dim]

    Tensor k_cache; // [n_layer, n_kv_heads, seq_len, head_dim] F16
    Tensor v_cache; // [n_layer, n_kv_heads, seq_len, head_dim] F16

    Tensor scores; // [n_heads, seq_len]
    Tensor context; // [n_heads, head_dim]

    Tensor mlp_gate; // [intermediate_size]
    Tensor mlp_up; // [intermediate_size]

    Tensor logits; // [vocab_size]
    Tensor probs; // [vocab_size]

    explicit InferenceState(const Config& config);

    void push_kv(size_t layer);
};
