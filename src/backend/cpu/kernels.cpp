#include <math.h>
#include "../../common/kernels.h"


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

    for (int i=0; i<n; i++){
        xout.data[i] = 0;
        for (int j=0; j<d; j++){
            xout.data[i] += w.get(i*d+j) * x.data[j];
        }
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





