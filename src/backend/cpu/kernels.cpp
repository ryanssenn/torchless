#include <math.h>
#include "../../../include/kernels.h"


// W (n,d) @ x (d,) = xout (n,)
void matmul(Tensor<float>& xout, Tensor<float>& w, Tensor<float>& x){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

    #pragma omp parallel for
    for (int i=0; i<n; i++){
        xout.data[i] = 0;
        for (int j=0; j<d; j++){
            xout.data[i] += w.data[i*d+j] * x.data[j];
        }
    }
}

void matmul(Tensor<float>& xout, Tensor<int8_t>& w, Tensor<float>& x){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.numel == d && xout.numel >= n && "matmul shape mismatch");

    // Group size for quantization (64 elements per scale)
    constexpr size_t GROUP_SIZE = 64;
    const size_t num_groups = d / GROUP_SIZE;
    
    #pragma omp parallel for
    for (int i=0; i<n; i++){
        float sum = 0.0f;
        size_t w_offset = i * d;
        
        // Process each group (64 elements share the same scale)
        for (size_t g=0; g<num_groups; g++){
            size_t group_start = g * GROUP_SIZE;
            size_t scale_idx = (w.scales.size() * (w_offset + group_start)) / w.numel;
            float scale = w.scales[scale_idx];
            
            // Direct dequantization: w.data[i] / scale (avoiding w.get() overhead)
            // Unroll loop for better performance
            float inv_scale = 1.0f / scale;
            size_t j = group_start;
            size_t group_end = group_start + GROUP_SIZE;
            if (group_end > d) group_end = d;
            
            // Process 4 elements at a time for better instruction-level parallelism
            for (; j + 4 <= group_end; j += 4){
                float dequant0 = w.data[w_offset + j] * inv_scale;
                float dequant1 = w.data[w_offset + j + 1] * inv_scale;
                float dequant2 = w.data[w_offset + j + 2] * inv_scale;
                float dequant3 = w.data[w_offset + j + 3] * inv_scale;
                sum += dequant0 * x.data[j] + dequant1 * x.data[j + 1] + 
                       dequant2 * x.data[j + 2] + dequant3 * x.data[j + 3];
            }
            
            // Handle remaining elements
            for (; j < group_end; j++){
                float dequant = w.data[w_offset + j] * inv_scale;
                sum += dequant * x.data[j];
            }
        }
        
        // Handle remaining elements if d is not a multiple of GROUP_SIZE
        size_t remaining_start = num_groups * GROUP_SIZE;
        for (size_t j = remaining_start; j < d; j++){
            // Need to get scale for this element
            sum += w.get(w_offset + j) * x.data[j];
        }
        
        xout.data[i] = sum;
    }
}


// x (,n) @ W (n,d) = xout (d,)
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& w){
    size_t n = w.shape[0];
    size_t d = w.shape[1];

    assert(x.shape[0] == n && xout.numel >= d && "matmul shape mismatch");

    for (int i=0; i<d; i++){
        xout.data[i] = 0;
        for (int j=0; j<n; j++){
            xout.data[i] += w.data[j*d+i] * x.data[j];
        }
    }
}


// e^x / sum(e^x)
// Here we compute the max and subtract it from each logit to avoid overflow,
// keeping the relative ratios unchanged (softmax is shift-invariant)
void softmax(Tensor<float>& xout, Tensor<float>& x){
    float maxv = x.max();
    float total = 0;
    for (int i=0; i<x.numel; i++){
        xout.data[i] = std::exp(x.data[i] - maxv);
        total += xout.data[i];
    }
    mul(xout, xout, 1/total);
}


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
// We perform 2D rotations each pair
// [x', y'] = [x*cosθ - y*sinθ, x*sinθ + y*cosθ]
// The same cos/sin embedding is reused for every head
// Inputs:
//   x: [n_heads, seq_len, head_dim] (q or k)
//   cos, sin: [seq_len, head_dim]

// Original RoPE rotates consecutive dim pairs (i, i+1), but Mistral uses a half-split layout?
// Each dim i is rotated with i + head_dim/2

// TODO: This only works 1 token at time (assumed seq_len = 1), cos/sin is only given for 1 pos
void rope(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& cos, Tensor<float>& sin){
    size_t n_heads = x.shape[0];
    //size_t seq_len = x.shape[1];
    size_t head_size = x.shape[1];
    size_t half = head_size/2;

    for (size_t h = 0; h<n_heads; h++){
        int start = h*x.strides[0];
        for (int i = start; i < start+half; i++){
            float xi = x.data[i];
            float yi = x.data[i+half];
            float c = cos.data[i - start];
            float s = sin.data[i - start];
            xout.data[i] = xi*c - yi*s;
            xout.data[i+half] = xi*s + yi*c;
        }
    }
}

// x / (1 + exp(-x))
void silu(Tensor<float>& xout, Tensor<float>& x){
    for (int i=0;i<x.numel;i++){
        xout.data[i] = x.data[i] / (1 + exp(-x.data[i]));
    }
}


// Not sure if I will be using all of those
float sum(Tensor<float>& x){
    float r = 0.0f;
    for (int i=0; i<x.numel; i++){
        r += x.data[i];
    }
    return r;
}

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] + y.data[i];
    }
}


void add(Tensor<float>& xout, Tensor<float>& x, float c){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] + c;
    }
}

void mul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y){
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] * y.data[i];
    }
}

void mul(Tensor<float>& xout, Tensor<float>& x, float c) {
    for (int i = 0; i < x.numel; i++) {
        xout.data[i] = x.data[i] * c;
    }
}

void pow(Tensor<float>& xout, Tensor<float>& x, int e){
    for (int i=0; i<x.numel; i++){
        xout.data[i] = pow(x.data[i], e);
    }
}

void sqrt(Tensor<float>& xout, Tensor<float>& x){
    for (int i=0; i<x.numel; i++){
        xout.data[i] = sqrt(x.data[i]);
    }
}





