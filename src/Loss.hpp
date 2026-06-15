// ===========================================================================
// Loss.hpp
// Categorical Cross-Entropy Loss + Softmax activation
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "Tensor.hpp"
#include "layers/Activations.hpp"    
#include <cmath>
#include <cassert>

namespace loss {

class CrossEntropyLoss {
private:
    Tensor last_softmax;  

public:
    
    float forward(const Tensor& logits, const Tensor& target) {
        assert(logits.n == target.n && "Batch size mismatch");
        assert(logits.c == target.c && "Number of classes mismatch");
        assert(logits.h == 1 && logits.w == 1 && 
               target.h == 1 && target.w == 1 && 
               "CrossEntropyLoss expects 1×1 spatial dimensions (after global pooling)");

        activations::Softmax softmax_act;
        last_softmax = softmax_act.forward(logits);

        float total_loss = 0.0f;
        const int batch = logits.n;
        const int channels = logits.c;

        for (int b = 0; b < batch; ++b) {
            for (int ch = 0; ch < channels; ++ch) {
                float prob = last_softmax(b, ch, 0, 0);
                total_loss -= target(b, ch, 0, 0) * std::log(prob + 1e-15f);
            }
        }

        return total_loss / static_cast<float>(batch);
    }

    Tensor backward(const Tensor& logits, const Tensor& target) {
        assert(!last_softmax.empty() && "Must call forward() before backward()");
        assert(logits.n == last_softmax.n && logits.c == last_softmax.c &&
               logits.h == last_softmax.h && logits.w == last_softmax.w &&
               "Cached softmax shape must match input logits");

        Tensor grad(logits.n, logits.c, logits.h, logits.w);

        const int batch = logits.n;
        const int channels = logits.c;

        for (int b = 0; b < batch; ++b) {
            for (int ch = 0; ch < channels; ++ch) {
                float prob = last_softmax(b, ch, 0, 0);
                float t = target(b, ch, 0, 0);
                grad(b, ch, 0, 0) = (prob - t) / static_cast<float>(batch);
            }
        }

        return grad;
    }
};

} 
