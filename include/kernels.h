//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"
#include <cmath>

void matmul(Tensor<float>& xout, Tensor<float>& w, Tensor<float>& x);
void matmul(Tensor<float>& xout, Tensor<int8_t>& w, Tensor<float>& x);
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& w);

void softmax(Tensor<float>& xout, Tensor<float>& x);
void rope(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& cos, Tensor<float>& sin);

void silu(Tensor<float>& xout, Tensor<float>& x);

float sum(Tensor<float>& x);

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);
void add(Tensor<float>& xout, Tensor<float>& x, float c);

void mul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);
void mul(Tensor<float>& xout, Tensor<float>& x, float c);

void pow(Tensor<float>& xout, Tensor<float>& x, int e);
void sqrt(Tensor<float>& xout, Tensor<float>& x);

#endif // MATH_OPS_H
