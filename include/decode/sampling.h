#pragma once

#include "model/inference_state.h"

#include <cstdint>

uint32_t sample_max(const InferenceState& infer);
uint32_t sample_multinomial(InferenceState& infer, float temp);
