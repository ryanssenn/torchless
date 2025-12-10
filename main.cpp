#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"
#include <random>
#include <chrono>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

uint32_t sample_max(InferenceState& infer){
    float max_val = infer.logits.data[0];
    size_t res = 0;
    for (size_t i = 0;i < infer.config.vocab_size; i++){
        if (infer.logits.data[i] > max_val){
            res = i;
            max_val = infer.logits.data[i];
        }
    }

    return res;
}

uint32_t sample_multinomial(InferenceState& infer, float temp){
    if (temp > 0) {
        for (int i=0; i<infer.logits.numel; i++) {
            infer.logits.data[i] /= temp;
        }
        softmax(infer.probs, infer.logits);
    } else {
        return sample_max(infer);
    }

    float r = distr(gen);
    float total = 0;

    for (int i=0; i<infer.probs.numel; i++){
        total += infer.probs.data[i];
        if (total >= r){
            return i;
        }
    }
    return infer.probs.numel - 1;
}

// Update generate to use sampling
template <typename T>
uint32_t generate(Model<T>& model, InferenceState& infer, size_t token){
    model.forward(infer, token);
    return sample_multinomial(infer, 0.0f);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters(model_path);

    InferenceState infer(params->config);
    Model<float> model(params);

    const std::string text = argv[2];

    std::vector<uint32_t> got = params->tokenizer.encode(text);

    for (int i=0;i<got.size()-1;i++){
        model.forward(infer, got[i]);
    }

    uint64_t start_time = get_timestamp_ms();

    uint32_t t = got[got.size()-1];
    int i = 0;
    for (;i<200;i++){
        t = generate(model, infer, t);

        if (t == 2){
            break;
        }

        std::cout << params->tokenizer.decode({t}) << std::flush;
    }

    std::cout << std::endl;

    uint64_t end_time = get_timestamp_ms();

    std::cout << "throughput: " << (i+1) / ((end_time - start_time) / 1000.0) << "tok/s" << std::endl;

    return 0;
}
