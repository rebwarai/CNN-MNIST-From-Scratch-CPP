// ===========================================================================
// Dropout.hpp
// Dropout layer for mnist-CNN-from-scratch
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "../Tensor.hpp"
#include "Layer.hpp"
#include <random>
#include <vector>
#include <cassert>
#include <string>

class Dropout : public Layer {
private:
    const float keep_prob;           
    const float train_scale;         

    bool is_training = true ;       

    Tensor mask;                    

    std::mt19937 rng;                
    std::bernoulli_distribution bernoulli;  

public:
    
    explicit Dropout(float kp = 0.05f)
        : keep_prob(1.0f - kp),
          train_scale(1.0f / kp),
          rng(std::random_device{}()),
          bernoulli(kp)
    {
        assert(keep_prob > 0.0f && keep_prob <= 1.0f);
    }

    void set_training(bool training) override {
        is_training = training;
    }

    Tensor forward(const Tensor& input) override {
        if (!is_training) {
            return input;
        }

        Tensor output = input;

        mask = Tensor(input.n, input.c, input.h, input.w);
        auto& mask_data = mask.data;

        for (size_t i = 0; i < mask_data.size(); ++i) {
            mask_data[i] = bernoulli(rng) ? 1.0f : 0.0f;
            output[i] *= mask_data[i] * train_scale;
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        if (!is_training) {
            return grad_output;
        }

        Tensor grad_input = grad_output;

        for (size_t i = 0; i < grad_input.size(); ++i) {
            grad_input[i] *= mask.data[i] * train_scale;
        }

        return grad_input;
    }

    std::vector<Tensor*> get_parameters() override {
        return {};
    }

    std::vector<Tensor> get_gradients() override {
        return {};
    }

    void zero_grad() override {}

    std::string name() const override {
        return "Dropout(" + std::to_string(1.0f - keep_prob) + ")";
    }
};
