// ===========================================================================
// Dense.hpp
// Fully Connected Layer for mnist-CNN-from-scratch
// author : @rebwar_ai
// ===========================================================================

#pragma once
#include "../Tensor.hpp"
#include "../Utils.hpp"

#include "Layer.hpp"
#include <vector>
#include <cassert>
#include <cmath>
#include <string>


class Dense : public Layer {
public:
    Tensor WT;  
    Tensor bT;      
    Tensor dWT;
    Tensor dbT;
    int in_features;
    int out_features;

    Dense(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        WT = Tensor(1,1,in_features,out_features);
        bT    = Tensor(1,1,1,out_features);

        WT.he(in_features);
        bT.zeros();
        dWT = U::zeros_like(WT);
        dbT = U::zeros_like(bT);
    }

    Tensor forward(const Tensor& x) override {
        input = x;
        int batch = x.n;
        Tensor out(batch, out_features, 1, 1);
        for (int b = 0; b < batch; ++b)
        {
            for (int o = 0; o < out_features; ++o) {
                float sum = bT(0,0,0,o);
                for (int i = 0; i < in_features; ++i)
                {
                    sum += x(b,i,0,0) * WT(0,0,i,o);
                } 
                out(b,o,0,0) = sum;
            }
        }
        return out;
    }

    Tensor backward(const Tensor& grad) override {
        int batch = grad.n;
        dWT.zeros();
        dbT.zeros();
        Tensor grad_input(batch, in_features, 1, 1);

        for (int b = 0; b < batch; ++b)
        {
            for (int o = 0; o < out_features; ++o) {
                dbT(0,0,0,o) += grad(b,o,0,0);
                for (int i = 0; i < in_features; ++i) {
                    dWT(0,0,i,o) += input(b,i,0,0) * grad(b,o,0,0);
                    grad_input(b,i,0,0) += WT(0,0,i,o) * grad(b,o,0,0);
                }
            }
        }

        for (auto &v : dWT.data)
        {
            v /= batch;
        }
        for (auto &v : dbT.data)
        {
            v /= batch;
        }

        return grad_input;
    }

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
        return "Dense(" + std::to_string(out_features) + ")";
    }

private:
    Tensor input;
};
