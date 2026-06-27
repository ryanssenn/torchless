#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

// Bump allocator for activation buffers. Tensor views point into arena memory;
// the arena (or caller) owns the lifetime.
struct Arena {
    static constexpr size_t kAlignment = 64;

    size_t BUFFER_SIZE;
    char* buffer;
    size_t offset = 0;

    explicit Arena(size_t buffer_size)
        : BUFFER_SIZE(buffer_size), buffer(new char[buffer_size]) {}

    ~Arena() {
        delete[] buffer;
    }

    void* allocate(size_t size) {
        size_t aligned = (offset + kAlignment - 1) & ~(kAlignment - 1);
        assert(aligned + size <= BUFFER_SIZE && "Arena out of memory");
        char* result = buffer + aligned;
        offset = aligned + size;
        return result;
    }

    void reset() {
        offset = 0;
    }
};
