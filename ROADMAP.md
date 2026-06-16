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
  Implements full byte-pair encoding (BPE) compatible with Mistral's vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

### Token Generation & Sampler
- [x] Basic text completion with greedy decoding
- [x] Multinomial sampling
- [x] Temperature scaling

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