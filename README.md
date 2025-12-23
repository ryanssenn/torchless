# Torchless

Torchless is a custom-built LLM inference engine written entirely from scratch. It currently runs [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU for local text completion.

<br>

![demo2](https://github.com/user-attachments/assets/1711dc3e-9ab2-4f73-8c35-b7ac3aabec55)

I am building this project to learn C++, performance engineering and LLM internals.

# Status

Next

- Rewrite of slow code sections
- Support [Ministral 3 3B](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)

Make it super fast with
- SIMD
- Quantization algorithms
- CUDA

## How it works

If you are new to LLM internals, an inference engine is essentially a loop that predicts the next word in a sequence, adds it to the history, and repeats. Here is the exact lifecycle of a prompt:

### Loading
Before we can run any math, we need the weights. The `export_mistral.py` script takes the complex Hugging Face folder structure and packs the weights into a single standardized binary file. The C++ engine loads this entire file into RAM at startup so the data is mapped and ready for computation.

### Tokenization
The model performs math on numbers, not strings. When you type a prompt like "Paris is", the `tokenizer` breaks it down using byte-pair Encoding (BPE). It looks up these chunks in the Mistral vocabulary and converts them into a list of integer IDs (e.g. [1, 782, 312]).

### Transformer Loop
We feed these IDs into the model one by one. The goal is to update a single vector, the `hidden_state`, as it passes through the network.

* **Embedding:** We take the input `token ID`and look up its specific floating-point vector in the embedding table. This turns a simple integer into a dense vector representing the token's initial semantic meaning.
* **Layers:** This state travels through 32 identical layers. In every layer, we first apply `RMSNorm` to stabilize the numbers. Then the state enters the `attention` module. It projects the state into `query`, `key`, and `value` vectors. The query "looks back" at the Keys of previous tokens to find relevant information (values). We apply `RoPE` (Rotary Positional Embeddings) so the model understands relative distance between words, then store the key and value in the `KV Cache`. This cache acts as the model's short-term memory, saving us from recalculating the history for every new word.
* **MLP:** Finally, the state goes through the `feedforward` module (a SwiGLU block). If Attention gathers context from the past, the MLP processes that information. It projects the vector to a higher dimension (14,336) to untangle complex relationships, applies a non-linear activation (SiLU), and projects it back down.

### Prediction
After 32 layers of processing, the final `hidden_state` holds the "meaning" of the next predicted token. We project this vector against the entire vocabulary to get `logits` raw confidence scores for all 32,000 possible next tokens. We run a `softmax` operation to turn these scores into probabilities and `sample` the result (either choosing the most likely token or picking randomly based on the probability distribution). We decode that ID back into text, print it, and feed it back into the transformer.


# Running

#### Download Mistral 7B v0.1, Torchless and nhlohmann JSON library

```
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/torchless.git
cd torchless
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o src/common/json.hpp
```

#### (Optional) Create Python virtual environment and download libraries

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Export the model with 8-bit quantization

```
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant f32
```

#### Compile project

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

#### Run
```
./torchless ../mistral.bin "Paris is the capital of"
```

If you run into issues that appear specific to your environment, feel free to open a GitHub issue.

# Resources

#### Some of the material that helped me learn the theory or guided me build the engine

##### ML Theory
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Andrej Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)

##### Systems Internals
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [C++ Vtables](https://shaharmike.com/cpp/vtable-part1/)
- [Andrew Chan - yalm](https://andrewkchan.dev/posts/yalm.html)
- [Arseny Kapoulkine - LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/)
- [Maxime Labonne - Quantize llama models](https://medium.com/data-science/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)

##### Reference Implementations
- [Hugging Face - Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Arseny Kapoulkine - calm](https://github.com/zeux/calm/tree/main)
- [Georgi Gerganov - llama.cpp + ggml](https://github.com/ggml-org/llama.cpp/)
- [Andrej Karpathy - llama2.c quantization](https://github.com/karpathy/llama2.c/blob/master/export.py)

# Roadmap

## Loading

### Model Loader
- [x] **Model binary converter** *(export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file, optionally applying quantization. It uses a JSON header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
- [x] **In-memory loader** *(src/loader/parameters.cpp)*  
  Memory-maps the binary, loads the config and provides direct tensor views.

### Tensor & Ops
- [x] **Tensor** *(src/common/tensor.cpp)*  
  Implements a strided view over memory supporting f32 and int8, with on-the-fly dequantization during compute
- [x] **Math operations** *(src/backend/cpu/kernels.cpp)*  
  Implementation of matmul, softmax and RoPE to be optimized later

## Text In, Tokens Out

### Tokenizer
- [x] **Tokenizer** *(src/tokenizer/tokenizer.cpp)*  
  Implements full byte-pair encoding (BPE) compatible with Mistral’s vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

### Token Generation & Sampler
- [x] Basic text completion with greedy decoding
- [x] Multinomial sampling
- [ ] Temperature scaling

### CLI I/O
- [ ] Build a terminal chat inferface

## Core Transformer
The architecture *(src/model/mistral/modules.cpp)* is broken into independent C++ structs using a shared inference state to manage memory and cache. The implementation encodes relative positions using rotary embeddings (RoPE), applies gated SwiGLU in the feed-forward layers, and utilizes grouped-query attention (GQA) which assigns multiple query heads to share a single key-value head pair.

### Inference State
- [x] Temporary memory and cache *(src/common/inference_state.h)* used to hold all intermediate tensors for a single token's computation during the forward pass

### Modules
- [x] **Embedding** - Looks up initial embedding from token and copies it to `infer.hidden_state`
- [x] **RMSNorm** - Initializes inverse frequencies based on rope theta and generates cosine/sine tables dynamically based on the current `infer.pos`
- [x] **Rotary Embedding** - precomputes inverse frequencies from rope_theta and fills cos/sin tensors for RoPE for each position
- [x] **Attention** - Projects to Q/K/V, applies rotary embeddings to Q/K, pushes to the KV cache, runs the grouped-query attention mechanism (reusing KV heads 4x), and projects the result.
- [x] **Feedforward MLP** - Implements the SwiGLU feedforward: linear projections + SiLU
- [x] **Layer** - Runs norm, attention, and MLP with residuals around each subblock
- [x] **Model** - Embeds input token and runs it through all decoder layers
- [x] **LM Head** - Projects the final `infer.hidden_state` onto the vocabulary dimension to populate `infer.logits`

### Parity Tests
- [x] Comprehensive validation in *(test/mistral)* of all inference components (tokenizer, modules, ops) by checking that their outputs match those produced by the Hugging Face Mistral implementation

## Gotta go fast

### Quantization
- [ ] Support fp8 with a cast during model export
- [x] Q8_K Per-group symmetric quantization - split tensor into groups, for each group, finds max abs value, computes scale and produces quantized weights

### CPU Multithreading
- [ ] Todo

### SIMD
- [ ] Todo

### Custom CUDA Kernels
- [ ] Todo
