// ===========================================================================
// Optimizer.hpp
// Adam optimizer for mnist-CNN-from-scratch
// Supports per-parameter updates, bias correction, and numerical stability
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "Tensor.hpp"
#include "Utils.hpp"
#include <vector>
#include <cmath>
#include <cassert>

namespace optim {

    class Adam {
    public:
        float lr;       
        float beta1;    
        float beta2;   
        float eps;     
        int t_step;         

        float lr_decay = 0.0f;   
        enum class DecayType { None, Step, Exponential } decay_type = DecayType::None;

        std::vector<Tensor> m; 
        std::vector<Tensor> v; 

        Adam(float learning_rate = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f)
            : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), t_step(0) {}

        void init(const std::vector<Tensor*>& params) {
            m.clear();
            v.clear();
            for (auto* p : params) {
                m.emplace_back(U::zeros_like(*p));
                v.emplace_back(U::zeros_like(*p));
            }
        }

        void set_decay(float decay, DecayType type = DecayType::Exponential) {
            lr_decay = decay;
            decay_type = type;
        }

        float current_lr() const {
            if (decay_type == DecayType::None) return lr;
            if (decay_type == DecayType::Step) return lr / (1.0f + lr_decay * t_step);
            if (decay_type == DecayType::Exponential) return lr * std::pow(1.0f - lr_decay, t_step);
            return lr; 
        }

        void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads) {
            assert(params.size() == grads.size());

            if (m.empty()) {
                init(params);
            }

            t_step += 1;
            float lr_t = current_lr();

            for (size_t i = 0; i < params.size(); ++i) {
                Tensor& param = *params[i];
                const Tensor& grad = grads[i];
                Tensor& mi = m[i];
                Tensor& vi = v[i];

                for (size_t j = 0; j < param.size(); ++j) {
                    mi[j] = beta1 * mi[j] + (1.0f - beta1) * grad[j];
                    vi[j] = beta2 * vi[j] + (1.0f - beta2) * grad[j] * grad[j];

                    float m_hat = mi[j] / (1.0f - std::pow(beta1, t_step));
                    float v_hat = vi[j] / (1.0f - std::pow(beta2, t_step));

                    param[j] -= lr_t * m_hat / (std::sqrt(v_hat) + eps);
                }
            }
        }
    };

} 
