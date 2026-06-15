// ===========================================================================
// Flatten.hpp
// Flatten layer for mnist-cnn-from-scratch
// Converts 4D tensor (N, C, H, W) -> 2D-like tensor (N, 1, 1, C*H*W)
// author : @rebwar_ai
// ===========================================================================

#pragma once
#include "../Tensor.hpp"
#include "Layer.hpp"

class Flatten : public Layer {
public:
    Tensor forward(const Tensor& x) override {
        input = x; 
        batch    = x.n;
        channels = x.c;
        height   = x.h;
        width    = x.w;
        features = channels * height * width;

        Tensor out(batch, features, 1, 1); 

        for (int b = 0; b < batch; ++b) {
            int idx = 0;
            for (int c = 0; c < channels; ++c)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        out(b, idx++, 0, 0) = x(b, c, y, w);
                    }
                }
            }
        }

        return out;
    }

    Tensor backward(const Tensor& grad) override {
        Tensor grad_input(batch, channels, height, width);
        for (int b = 0; b < batch; ++b) {
            int idx = 0;
            for (int c = 0; c < channels; ++c)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        grad_input(b, c, y, w) = grad(b, idx++, 0, 0);
                    }
                }
            }
        }
        return grad_input;
    }

    std::string name() const override {
        return "Flatten(" + std::to_string(this->features) + ")";
    }

    void zero_grad() override {}

private:
    Tensor input;
    int batch = 0, channels = 0, height = 0, width = 0, features = 0;
};
