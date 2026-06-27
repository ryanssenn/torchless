#pragma once

#include "common/arena.h"
#include "common/dtype.h"
#include "common/shape.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

// Non-owning view into memory owned by Arena (activations) or mmap (weights).
// Tensor never allocates or frees storage.
struct Tensor {
    DType dtype = DType::F32;
    void* data = nullptr;
    size_t numel = 0;

    uint8_t ndim = 0;
    std::array<size_t, 4> shape{};
    std::array<size_t, 4> strides{};

    // INT8 weights only; empty for F32/F16
    std::vector<float> scales;

    // Wrap existing memory with the given dtype and dimensions.
    static Tensor from_ptr(void* data, DType dtype, std::initializer_list<size_t> dims);
    // Wrap existing memory with per-element INT8 dequant scales.
    static Tensor from_ptr(void* data, DType dtype, std::vector<float> scales, std::initializer_list<size_t> dims);
    // Wrap existing memory using a pre-built Shape.
    static Tensor from_ptr(void* data, DType dtype, const Shape& shape);
    // Wrap existing memory with scales using a pre-built Shape.
    static Tensor from_ptr(void* data, DType dtype, std::vector<float> scales, const Shape& shape);

    // Allocate fresh storage from arena and return a view into it.
    static Tensor alloc(Arena& arena, DType dtype, std::initializer_list<size_t> dims);

    // Element size in bytes for this tensor's dtype.
    size_t type_size() const { return dtype_size(dtype); }
    // Total storage size in bytes.
    size_t byte_size() const { return numel * type_size(); }
    // True when elements are laid out contiguously in row-major order.
    bool is_contiguous() const { return true; }

    // Typed data pointer; asserts dtype is F32.
    float* f32() const;
    // Typed data pointer; asserts dtype is INT8.
    int8_t* i8() const;
    // Typed data pointer; asserts dtype is F16.
    __fp16* f16() const;

    // Deprecated aliases for f32/i8/f16.
    float* as_f32() const { return f32(); }
    int8_t* as_i8() const { return i8(); }
    __fp16* as_f16() const { return f16(); }

    // Read element i as F32, converting from the stored dtype if needed.
    float get(size_t i) const;
    // Maximum element value; F32 only.
    float max() const;

    // Copy same-dtype elements from src into this tensor.
    void copy_from(const Tensor& src);

    // Subtensor view at the given leading indices.
    Tensor at(std::initializer_list<size_t> idx) const;
    // View the same data with a different shape; numel must match.
    Tensor reshape(std::initializer_list<size_t> new_dims) const;
    // Shrink a 1-D view to its first len elements.
    Tensor view_prefix(size_t len) const;
    // Shrink the leading dimension of a 2-D+ view to rows rows.
    Tensor view_rows(size_t rows) const;
};

// Allocate an F32 tensor and copy arr into it (test helpers).
Tensor make_f32_tensor(Arena& arena, const std::vector<float>& arr, std::initializer_list<size_t> dims);
