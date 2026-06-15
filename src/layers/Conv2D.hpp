// ===========================================================================
// Conv2D.hpp
// Convolutional layer for mnist-CNN-from-scratch
// Supports stride, padding, forward + full backward
// author : @rebwar_ai
// ===========================================================================
 
#pragma once

#include "../Tensor.hpp"
#include "Layer.hpp"
#include <vector>
#include <cassert>
#include <cmath>

class Conv2D : public Layer {
public:
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
    int padding;

    Tensor WT;  
    Tensor bT;   

    Tensor dWT;
    Tensor dbT;

    Conv2D(int in_c, int out_c, int k, int s = 1, int p = 0)
        : in_channels(in_c),
          out_channels(out_c),
          kernel(k),
          stride(s),
          padding(p)
    {
        WT  = Tensor(out_channels, in_channels, kernel, kernel);
        bT  = Tensor(1, out_channels, 1, 1);
        dWT = Tensor(out_channels, in_channels, kernel, kernel);
        dbT = Tensor(1, out_channels, 1, 1);

        int fan_in  = in_channels * kernel * kernel;
        
        WT.he(fan_in);
        bT.zeros();
    }

    // ==========================
    // Forward
    // ==========================
    Tensor forward(const Tensor& X) override {
        input = X;

        int N = X.n;
        int C = X.c;
        int H = X.h;
        int W = X.w;

        assert(C == in_channels);

        H_out = (H + 2 * padding - kernel) / stride + 1;
        W_out = (W + 2 * padding - kernel) / stride + 1;
        
        Tensor out(N, out_channels, H_out, W_out);

        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oy = 0; oy < H_out; ++oy) {
                    for (int ox = 0; ox < W_out; ++ox) {

                        float sum = bT(0, oc, 0, 0);

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int ky = 0; ky < kernel; ++ky) {
                                for (int kx = 0; kx < kernel; ++kx) {

                                    int iy = oy * stride + ky - padding;
                                    int ix = ox * stride + kx - padding;

                                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                        sum += input(n, ic, iy, ix)
                                             * WT(oc, ic, ky, kx);
                                    }
                                }
                            }
                        }

                        out(n, oc, oy, ox) = sum;
                    }
                }
            }
        }
        return out;
    }

    // ==========================
    // Backward
    // ==========================
    Tensor backward(const Tensor& grad) override {
        int N = input.n;
        int H = input.h;
        int W = input.w;

        dWT.zeros();
        dbT.zeros();

        Tensor dX(N, in_channels, H, W);
        dX.zeros();

        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oy = 0; oy < H_out; ++oy) {
                    for (int ox = 0; ox < W_out; ++ox) {

                        float g = grad(n, oc, oy, ox);
                        dbT(0, oc, 0, 0) += g;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int ky = 0; ky < kernel; ++ky) {
                                for (int kx = 0; kx < kernel; ++kx) {

                                    int iy = oy * stride + ky - padding;
                                    int ix = ox * stride + kx - padding;

                                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                        dWT(oc, ic, ky, kx) += input(n, ic, iy, ix) * g;
                                        dX(n, ic, iy, ix) += WT(oc, ic, ky, kx) * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        
        for (size_t i = 0; i < dWT.size(); ++i) dWT[i] /= N;
        for (size_t i = 0; i < dbT.size(); ++i) dbT[i] /= N;

        return dX;
    }

    // ==========================
    // Optimizer helpers
    // ==========================
    void zero_grad() override {
        dWT.zeros();
        dbT.zeros();
    }

    std::vector<Tensor*> get_parameters() override {
        return { &WT, &bT };
    }

    std::vector<Tensor> get_gradients() override {
        return { dWT, dbT };
    }

    void save(std::ostream& os) const override {

        os.write(reinterpret_cast<const char*>(WT.data.data()),
                 WT.size() * sizeof(float));
        os.write(reinterpret_cast<const char*>(bT.data.data()),
                 bT.size() * sizeof(float));
    }

    void load(std::istream& is) override {
        is.read(reinterpret_cast<char*>(WT.data.data()),
                WT.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(bT.data.data()),
                bT.size() * sizeof(float));
    }

    std::string name() const override {
        return "Conv2D(" + std::to_string(out_channels) + ")";
    }


private:
    Tensor input;
    int H_out = 0;
    int W_out = 0;
};
