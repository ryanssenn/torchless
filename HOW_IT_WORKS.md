# How it works

If you are new to LLM internals, an inference engine is essentially a loop that predicts the next word in a sequence, adds it to the history, and repeats. Here is the full lifecycle:

## Loading
Before we can run any math, we need the weights. The `export_mistral.py` script takes the complex Hugging Face folder structure and packs the weights into a single standardized binary file. The C++ engine loads this entire file into RAM at startup so the data is mapped and ready for computation.

## Tokenization
The model performs math on numbers, not strings. When you type a prompt like "Paris is", the `tokenizer` breaks it down using byte-pair Encoding (BPE). It looks up these chunks in the Mistral vocabulary and converts them into a list of integer IDs (e.g. [1, 782, 312]).

## Transformer Loop
We feed these IDs into the model one by one. The goal is to update a single vector, the `hidden_state`, as it passes through the network.

* **Embedding:** We take the input `token ID`and look up its specific floating-point vector in the embedding table. This turns a simple integer into a dense vector representing the token's initial semantic meaning.
* **Layers:** This state travels through 32 identical layers. In every layer, we first apply `RMSNorm` to stabilize the numbers. Then the state enters the `attention` module. It projects the state into `query`, `key`, and `value` vectors. The query "looks back" at the Keys of previous tokens to find relevant information (values). We apply `RoPE` (Rotary Positional Embeddings) so the model understands relative distance between words, then store the key and value in the `KV Cache`. This cache acts as the model's short-term memory, saving us from recalculating the history for every new word.
* **MLP:** Finally, the state goes through the `feedforward` module (a SwiGLU block). If Attention gathers context from the past, the MLP processes that information. It projects the vector to a higher dimension (14,336) to untangle complex relationships, applies a non-linear activation (SiLU), and projects it back down.

## Prediction
After 32 layers of processing, the final `hidden_state` holds the "meaning" of the next predicted token. We project this vector against the entire vocabulary to get `logits` raw confidence scores for all 32,000 possible next tokens. We run a `softmax` operation to turn these scores into probabilities and `sample` the result (either choosing the most likely token or picking randomly based on the probability distribution). We decode that ID back into text, print it, and feed it back into the transformer.
