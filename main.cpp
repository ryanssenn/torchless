#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

uint32_t sample_max(InferenceState& infer) {
	float max_val = infer.logits.data[0];
	size_t res = 0;
	for (size_t i = 0; i < infer.config.vocab_size; i++) {
		if (infer.logits.data[i] > max_val) {
			res = i;
			max_val = infer.logits.data[i];
		}
	}

	return res;
}

uint32_t sample_multinomial(InferenceState& infer, float temp) {
	if (temp > 0) {
		for (int i = 0; i < infer.logits.numel; i++) {
			infer.logits.data[i] /= temp;
		}
		softmax(infer.probs, infer.logits);
	}
	else {
		return sample_max(infer);
	}

	float r = distr(gen);
	float total = 0;

	for (int i = 0; i < infer.probs.numel; i++) {
		total += infer.probs.data[i];
		if (total >= r) {
			return i;
		}
	}
	return infer.probs.numel - 1;
}

// Update generate to use sampling
template <typename T>
uint32_t generate(Model<T>& model, InferenceState& infer, size_t token) {
	model.forward(infer, token);
	return sample_multinomial(infer, 0.0f);
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>" << std::endl;
		return 1;
	}

	std::string model_path = argv[1];

	auto params = std::make_shared<Parameters>();
	params->load_parameters(model_path);

	const std::string text = argv[2];
	std::vector<uint32_t> got = params->tokenizer.encode(text);

	if (params->config.quant == "int8") {
		InferenceState infer(params->config);
		Model<int8_t> model(params);

		for (size_t i = 0; i + 1 < got.size(); i++) {
			model.forward(infer, got[i]);
		}

		uint32_t t = got.back();
		for (int i = 0; i < 50; i++) {
			t = generate<int8_t>(model, infer, t);
			std::cout << params->tokenizer.decode({ t }) << std::flush;
		}
		std::cout << std::endl;
	}
	else {
		InferenceState infer(params->config);
		Model<float> model(params);

		for (size_t i = 0; i + 1 < got.size(); i++) {
			model.forward(infer, got[i]);
		}

		uint32_t t = got.back();
		for (int i = 0; i < 50; i++) {
			t = generate<float>(model, infer, t);
			std::cout << params->tokenizer.decode({ t }) << std::flush;
		}
		std::cout << std::endl;
	}

	return 0;
}
