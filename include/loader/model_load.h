#pragma once

#include <cstdint>
#include <unordered_map>
#include <string>
#include "common/fp16.h"
#include "common/tensor.h"
#include "tokenizer/qwen_tokenizer.h"
#include <variant>

namespace model_format {

constexpr char MAGIC[4] = {'M', 'O', 'G', '\0'};
constexpr uint32_t FORMAT_VERSION = 2;

enum class DType : uint8_t {
    F32 = 0,
    INT8 = 1,
    F16 = 2,
};

enum class ConfigValueType : uint8_t {
    STRING = 0,
    UINT32 = 1,
    FLOAT32 = 2,
};

constexpr size_t FILE_PREFIX_SIZE = 16; // magic + version + header_size

} // namespace model_format

struct Config {
    std::string architecture;
    size_t hidden_size;
    size_t intermediate_size;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t vocab_size;
    size_t head_dim;
    size_t sliding_window;
    size_t max_position_embeddings;
    float rope_theta;
    float norm_eps;
    std::string quant;
    bool tie_word_embeddings = false;
    uint32_t bos_token_id = 0;
    uint32_t eos_token_id = 0;
};

class BinaryReader;

using WeightTensor = std::variant<Tensor<float>, Tensor<int8_t>, Tensor<fp16_t>>;

struct ModelLoad {
    Config config;
    QwenTokenizer tokenizer;

    std::unordered_map<std::string, WeightTensor> global_weights;
    std::vector<std::unordered_map<std::string, WeightTensor>> layer_weights;

    static void* map_file(int fd, size_t& size);
    void load_tensor(std::unordered_map<std::string, WeightTensor>& m, char* p, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset, uint64_t scale_offset, uint32_t scale_size);

    void load_config(BinaryReader& reader);
    void load_weights(char* p, BinaryReader& reader);
    void load(const std::string& path);

    template<typename T>
    Tensor<T> get_tensor(int layer, const std::string& name);
};

inline bool is_f16(const std::string& quant) {
    return quant == "f16";
}
