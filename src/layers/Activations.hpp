// ===========================================================================
// Activations.hpp
// Activation functions for mnist-CNN-from-scratch
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "../Tensor.hpp"
#include "../Utils.hpp"
#include "Layer.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>
#include <vector>

namespace activations {

    // ==========================
    // ReLU Activation
    // ==========================
    class ReLU : public Layer {
    private:
        Tensor last_input;  

    public:
        Tensor forward(const Tensor& input) override {
            
            last_input = U::copy(input);  

            Tensor output = U::copy(input);

            const size_t total = input.size();
            for (size_t i = 0; i < total; ++i) {
                output[i] = std::max(0.0f, input[i]);
            }

            return output;
        }

        Tensor backward(const Tensor& grad_output) override {
            
            assert(grad_output.size() == last_input.size() && "Gradient shape mismatch");

            Tensor grad_input = U::copy(grad_output);

            const size_t total = last_input.size();
            for (size_t i = 0; i < total; ++i) {
                if (last_input[i] <= 0.0f) {
                    grad_input[i] = 0.0f;
                }
            }

            return grad_input;
        }

        std::string name() const override {
            return "ReLU()";
        }

        void zero_grad() override {}

        std::vector<Tensor*> get_parameters() override { return {}; }

        std::vector<Tensor> get_gradients() override { return {}; }
    };


    // ==========================
    // Softmax Activation
    // ==========================
    class Softmax : public Layer {
    private:
        Tensor last_output;

    public:
        Tensor forward(const Tensor& input) override {
            last_output = U::copy(input); 

            int N = input.n;
            int C = input.c;
            int H = input.h;
            int W = input.w;

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {

                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int c = 0; c < C; ++c) {
                            max_val = std::max(max_val, input(n, c, h, w));  
                        }

                        float sum_exp = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            float exp_val = std::exp(input(n, c, h, w) - max_val);
                            last_output(n, c, h, w) = exp_val;
                            sum_exp += exp_val;  
                        }

                        if (sum_exp > 0.0f) {
                            for (int c = 0; c < C; ++c) {
                                last_output(n, c, h, w) /= sum_exp;  
                            }
                        }
                    }
                }
            }

            return last_output;
        }

        Tensor backward(const Tensor& grad_output) override {
            // This method not used
           
            assert(last_output.size() == grad_output.size() && "Shape mismatch in softmax backward");

            Tensor grad_input = U::copy(grad_output);

            int N = last_output.n;
            int C = last_output.c;
            int H = last_output.h;
            int W = last_output.w;

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float dot = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            dot += last_output(n, c, h, w) * grad_output(n, c, h, w);
                        }

                        for (int c = 0; c < C; ++c) {
                            float s = last_output(n, c, h, w); 
                            grad_input(n, c, h, w) = s * (grad_output(n, c, h, w) - dot);
                        }
                    }
                }
            }

            return grad_input;
        }

        std::string name() const override {
            return "Softmax()";
        }

        void zero_grad() override {}

        std::vector<Tensor*> get_parameters() override { return {}; }

        std::vector<Tensor> get_gradients() override { return {}; }
    };

} 
