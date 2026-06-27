#include "common/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstring>

// Build a Tensor view from raw fields and optional INT8 scales.
static Tensor make_tensor(void* data, DType dtype, const Shape& shape, std::vector<float> scales = {}) {
    Tensor t;
    t.dtype = dtype;
    t.data = data;
    t.ndim = shape.ndim;
    t.shape = shape.dims;
    t.numel = numel_from_shape(shape);
    init_strides(t.strides, shape);
    t.scales = std::move(scales);
    return t;
}

// Wrap existing memory with the given dtype and dimensions.
Tensor Tensor::from_ptr(void* data, DType dtype, std::initializer_list<size_t> dims) {
    return from_ptr(data, dtype, Shape::from_dims(dims));
}

// Wrap existing memory with per-element INT8 dequant scales.
Tensor Tensor::from_ptr(void* data, DType dtype, std::vector<float> scales, std::initializer_list<size_t> dims) {
    return from_ptr(data, dtype, std::move(scales), Shape::from_dims(dims));
}

// Wrap existing memory using a pre-built Shape.
Tensor Tensor::from_ptr(void* data, DType dtype, const Shape& shape) {
    return make_tensor(data, dtype, shape);
}

// Wrap existing memory with scales using a pre-built Shape.
Tensor Tensor::from_ptr(void* data, DType dtype, std::vector<float> scales, const Shape& shape) {
    return make_tensor(data, dtype, shape, std::move(scales));
}

// Allocate fresh storage from arena and return a view into it.
Tensor Tensor::alloc(Arena& arena, DType dtype, std::initializer_list<size_t> dims) {
    Shape shape = Shape::from_dims(dims);
    size_t bytes = numel_from_shape(shape) * dtype_size(dtype);
    void* ptr = arena.allocate(bytes);
    return make_tensor(ptr, dtype, shape);
}

// Allocate an F32 tensor and copy arr into it (test helpers).
Tensor make_f32_tensor(Arena& arena, const std::vector<float>& arr, std::initializer_list<size_t> dims) {
    Tensor t = Tensor::alloc(arena, DType::F32, dims);
    assert(arr.size() == t.numel);
    std::memcpy(t.data, arr.data(), t.byte_size());
    return t;
}

// Typed data pointer; asserts dtype is F32.
float* Tensor::f32() const {
    assert(dtype == DType::F32);
    return static_cast<float*>(data);
}

// Typed data pointer; asserts dtype is INT8.
int8_t* Tensor::i8() const {
    assert(dtype == DType::INT8);
    return static_cast<int8_t*>(data);
}

// Typed data pointer; asserts dtype is F16.
__fp16* Tensor::f16() const {
    assert(dtype == DType::F16);
    return static_cast<__fp16*>(data);
}

// Copy same-dtype elements from src into this tensor.
void Tensor::copy_from(const Tensor& src) {
    assert(dtype == src.dtype);
    assert(numel == src.numel);
    std::memcpy(data, src.data, byte_size());
}

// Subtensor view at the given leading indices.
Tensor Tensor::at(std::initializer_list<size_t> idx) const {
    assert(idx.size() <= ndim && "Too many indices for tensor");
    char* new_data = static_cast<char*>(data);

    uint8_t i = 0;
    for (size_t v : idx) {
        assert(v < shape[i] && "Index out of range");
        new_data += strides[i] * v * type_size();
        i++;
    }

    Shape new_shape;
    new_shape.ndim = ndim - i;
    for (uint8_t j = 0; j < new_shape.ndim; j++) {
        new_shape.dims[j] = shape[i + j];
    }
    return make_tensor(new_data, dtype, new_shape, scales);
}

// View the same data with a different shape; numel must match.
Tensor Tensor::reshape(std::initializer_list<size_t> new_dims) const {
    Shape new_shape = Shape::from_dims(new_dims);
    size_t new_numel = numel_from_shape(new_shape);
    assert(new_numel == numel && "Reshape size mismatch");
    return make_tensor(data, dtype, new_shape, scales);
}

// Shrink a 1-D view to its first len elements.
Tensor Tensor::view_prefix(size_t len) const {
    assert(ndim == 1);
    assert(len <= shape[0]);
    Shape new_shape = Shape::from_dims({len});
    return make_tensor(data, dtype, new_shape, scales);
}

// Shrink the leading dimension of a 2-D+ view to rows rows.
Tensor Tensor::view_rows(size_t rows) const {
    assert(ndim >= 2);
    assert(rows <= shape[0]);
    Shape new_shape;
    new_shape.ndim = ndim;
    new_shape.dims[0] = rows;
    for (uint8_t i = 1; i < ndim; i++) {
        new_shape.dims[i] = shape[i];
    }
    size_t new_numel = rows;
    for (uint8_t i = 1; i < ndim; i++) {
        new_numel *= shape[i];
    }
    Tensor t = make_tensor(data, dtype, new_shape, scales);
    t.numel = new_numel;
    return t;
}

// Maximum element value; F32 only.
float Tensor::max() const {
    assert(dtype == DType::F32);
    float result = f32()[0];
    for (size_t i = 1; i < numel; i++) {
        result = std::max(result, f32()[i]);
    }
    return result;
}

// Read element i as F32, converting from the stored dtype if needed.
float Tensor::get(size_t i) const {
    switch (dtype) {
        case DType::F32:
            return f32()[i];
        case DType::F16:
            return static_cast<float>(f16()[i]);
        case DType::INT8:
            assert(false && "INT8 dequant is done in kernels");
            return 0.f;
    }
    return 0.f;
}
