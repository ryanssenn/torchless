#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct ModelLoad;
struct InferenceState;

void run_perplexity(std::shared_ptr<ModelLoad> params, InferenceState& infer, const std::vector<uint32_t>& tokens);
