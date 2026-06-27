#include "context.h"
#include <fstream>
#include <sstream>
#include <limits>

std::vector<TestCase> tests;

std::shared_ptr<ModelLoad> get_model(){
    static bool init = false;

    static std::shared_ptr<ModelLoad> params = std::make_shared<ModelLoad>();

    if (init){
        return params;
    }

    std::string model_path = "mistral-7B-Q8F16.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../mistral-7B-Q8F16.mog";
    }

    params->load(model_path);
    init = true;

    return params;
}

void load_expected_values(){
    auto load_file = [](const std::string& path) {
        if (!std::ifstream(path).good()) {
            return;
        }

        std::ifstream f(path);
        std::string line, name;

        while (std::getline(f, name)){
            if (name == "" || name[0] == '#') continue;
            std::getline(f, line);
            std::stringstream ss(line);
            float x;
            size_t i = 0;
            std::vector<float> data;
            while (ss >> x){
                data.push_back(x);
                i++;
            }
            expected.emplace(name, make_f32_tensor(arena, data, {i}));
        }
    };

    load_file("test/mistral/expected.txt");
    load_file("test/mistral/logits_expected.txt");
}

bool has_logits_golden(){
    return expected.find("logits_sky_step0_top10_ids") != expected.end();
}

TopK get_topk(const Tensor& logits, size_t k){
    TopK result;
    const float* data = logits.f32();
    std::vector<bool> used(logits.numel, false);

    for (size_t n = 0; n < k; n++){
        float best = -std::numeric_limits<float>::infinity();
        size_t best_i = 0;

        for (size_t i = 0; i < logits.numel; i++){
            if (!used[i] && data[i] > best){
                best = data[i];
                best_i = i;
            }
        }

        used[best_i] = true;
        result.ids.push_back(static_cast<uint32_t>(best_i));
        result.vals.push_back(best);
    }

    return result;
}

static float equals_atol(){
    if (is_q8f16(get_model()->config.quant)) {
        return 1.1e-1f;
    }
    return 5e-2f;
}

bool equals(float x, float y){
    return std::fabs(x - y) < equals_atol();
}

bool equals(const Tensor& x, const Tensor& y){
    return equals(x, y, equals_atol());
}

bool equals(const Tensor& x, const Tensor& y, float atol){
    if (x.numel != y.numel){
        return false;
    }

    const float* x_data = x.f32();
    const float* y_data = y.f32();
    for (size_t i=0; i<x.numel; i++){
        if (std::fabs(x_data[i] - y_data[i]) >= atol){
            std::cout << "Mismatch at pos " << i
                      << ": expected " << y_data[i]
                      << ", got " << x_data[i] << std::endl;
            return false;
        }
    }

    return true;
}
