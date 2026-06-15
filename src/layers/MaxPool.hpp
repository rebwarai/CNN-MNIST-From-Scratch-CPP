// ===========================================================================
// MaxPool.hpp
// Max pooling layer for mnist-cnn-from-scratch
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "../Tensor.hpp"
#include "Layer.hpp"
#include <vector>
#include <cassert>

class MaxPool : public Layer {
public:
    int kernel;   
    int stride; 

    MaxPool(int k, int s) : kernel(k), stride(s) {}

    Tensor forward(const Tensor& x) override {
        input = x; 

        n = x.n;
        c = x.c;
        h_in = x.h;
        w_in = x.w;

        h_out = (h_in - kernel) / stride + 1;
        w_out = (w_in - kernel) / stride + 1;

        Tensor out(n, c, h_out, w_out);
        mask.clear();
        mask.resize(n * c * h_out * w_out, std::make_pair(0,0));

        for (int b = 0; b < n; ++b) {
            for (int ch = 0; ch < c; ++ch) {
                for (int y = 0; y < h_out; ++y) {
                    for (int x_pos = 0; x_pos < w_out; ++x_pos) {
                        float max_val = -1e9f;
                        int max_y = -1, max_x = -1;

                        for (int ky = 0; ky < kernel; ++ky) {
                            for (int kx = 0; kx < kernel; ++kx) {
                                int yy = y * stride + ky;
                                int xx = x_pos * stride + kx;
                                float val = input(b, ch, yy, xx);
                                if (val > max_val) {
                                    max_val = val;
                                    max_y = yy;
                                    max_x = xx;
                                }
                            }
                        }

                        out(b, ch, y, x_pos) = max_val;
                        mask[index_mask(b,ch,y,x_pos)] = {max_y, max_x};
                    }
                }
            }
        }

        return out;
    }

    Tensor backward(const Tensor& grad) override {
        Tensor grad_input(n, c, h_in, w_in);
        grad_input.zeros();

        for (int b = 0; b < n; ++b) {
            for (int ch = 0; ch < c; ++ch) {
                for (int y = 0; y < h_out; ++y) {
                    for (int x_pos = 0; x_pos < w_out; ++x_pos) {
                        auto [yy, xx] = mask[index_mask(b,ch,y,x_pos)];
                        grad_input(b,ch,yy,xx) += grad(b,ch,y,x_pos);
                    }
                }
            }
        }

        return grad_input;
    }

    std::string name() const override {
        return "MaxPool(" + std::to_string(this->kernel) +","+ std::to_string(this->stride) + ")";
    }


    void zero_grad() override {}

private:
    Tensor input;
    int n, c, h_in, w_in, h_out, w_out;

    std::vector<std::pair<int,int>> mask;

    size_t index_mask(int b, int ch, int y, int x_pos) const noexcept {
        return static_cast<size_t>(((b * c + ch) * h_out + y) * w_out + x_pos);
    }
};
