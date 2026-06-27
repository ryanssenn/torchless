#pragma once

#include "common/tensor.h"

void matmul(Tensor& xout, const Tensor& w, const Tensor& x);
void row_matmul(Tensor& xout, const Tensor& x, const Tensor& w);

void softmax(Tensor& xout, const Tensor& x);
void rope(Tensor& xout, const Tensor& x, const Tensor& cos, const Tensor& sin);
void silu(Tensor& xout, const Tensor& x);
void add(Tensor& xout, const Tensor& x, const Tensor& y);
void mul(Tensor& xout, const Tensor& x, const Tensor& y);
void mul(Tensor& xout, const Tensor& x, float c);
