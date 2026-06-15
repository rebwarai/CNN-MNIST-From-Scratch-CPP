// ===========================================================================
// Layer.hpp
// Virtual base class for all neural network layers
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "../Tensor.hpp"
#include <vector>
#include <string>

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual Tensor backward(const Tensor& grad_output) = 0;

    virtual void zero_grad() = 0;

    virtual void set_training(bool) {}


    virtual std::vector<Tensor*> get_parameters() {
        return {};
    }

    virtual std::vector<Tensor> get_gradients() {
        return {};
    }

    virtual void save(std::ostream&) const {}
    virtual void load(std::istream&) {}
    virtual std::string name() const {
        return "Layer"; 
    }
};