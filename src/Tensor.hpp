// ===========================================================================
// Tensor.hpp
// 4D Tensor class for mnist-CNN-from-scratch
// Format: NCHW (Batch, Channels, Height, Width)
// author : @rebwar_ai
// ===========================================================================
 
#pragma once

#include <vector>      
#include <array>        
#include <iostream>    
#include <iomanip>      
#include <algorithm>    
#include <random>       
#include <cmath>        
#include <cassert>      
#include <cstddef>      
#include <string>

class Tensor {
public:
    // Shape: N (batch), C (channels), H (height), W (width)
    int n = 0;  // batch size
    int c = 0;  // channels
    int h = 0;  // height
    int w = 0;  // width

    std::vector<float> data;

    Tensor() = default;

    explicit Tensor(int batch, int channels, int height, int width)
        : n(batch), c(channels), h(height), w(width),
          data(static_cast<size_t>(batch) * 
                static_cast<size_t>(channels) * 
                static_cast<size_t>(height) * 
                static_cast<size_t>(width), 0.0f) {}

    Tensor(int batch, int channels, int height, int width, std::vector<float>&& src)
        : n(batch), c(channels), h(height), w(width), data(std::move(src)) {
        const size_t expected = static_cast<size_t>(batch) *
                                static_cast<size_t>(channels) *
                                static_cast<size_t>(height) *
                                static_cast<size_t>(width);

        assert(expected == data.size() && "Data size must match shape (n*c*h*w).");
    }

    [[nodiscard]] size_t size() const noexcept { return data.size(); }
    [[nodiscard]] std::array<int, 4> shape() const noexcept { return {n, c, h, w}; }
    [[nodiscard]] bool empty() const noexcept { return data.empty(); }

    float& operator()(int bn, int ch, int y, int x) {
        assert(bn >= 0 && bn < n && "Batch index out of bounds");
        assert(ch >= 0 && ch < c && "Channel index out of bounds");
        assert(y >= 0 && y < h && "Row index out of bounds");
        assert(x >= 0 && x < w && "Column index out of bounds");
        return data[index(bn, ch, y, x)];
    }

    float operator()(int bn, int ch, int y, int x) const {
        assert(bn >= 0 && bn < n && "Batch index out of bounds");
        assert(ch >= 0 && ch < c && "Channel index out of bounds");
        assert(y >= 0 && y < h && "Row index out of bounds");
        assert(x >= 0 && x < w && "Column index out of bounds");
        return data[index(bn, ch, y, x)];
    }


    float& operator[](size_t i) 
    { 
        assert(i < data.size() && "Index out of bounds"); 
        return data[i]; 
    }
    const float& operator[](size_t i) const 
    { 
        assert(i < data.size() && "Index out of bounds"); 
        return data[i]; 
    }

    void zeros() noexcept { 
        std::fill(data.begin(), data.end(), 0.0f); 
    }
    void fill(float value) noexcept { 
        std::fill(data.begin(), data.end(), value); 
    }

    void random_uniform(float range = 0.1f) {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(-range, range);
        for (auto& v : data) 
        {
            v = dis(gen);
        }
    }

    void he(int in_features) {
        float stddev = std::sqrt(2.0f / static_cast<float>(in_features));
        static std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dis(0.0f, stddev);
        for (auto& v : data)
        {
            v = dis(gen);
        }
    }

    void xavier(int in_features, int out_features) {
        float limit = std::sqrt(6.0f / (in_features + out_features));
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(-limit, limit);
        for (auto& v : data)
        {
            v = dis(gen);
        }
    }

    void reshape(int new_n, int new_c, int new_h, int new_w) {
        const size_t old_count = static_cast<size_t>(n) * 
                                static_cast<size_t>(c) * 
                                static_cast<size_t>(h) * 
                                static_cast<size_t>(w);

        const size_t new_count = static_cast<size_t>(new_n) * 
                                static_cast<size_t>(new_c) * 
                                static_cast<size_t>(new_h) * 
                                static_cast<size_t>(new_w);

        assert(new_count == old_count && "Total number of elements must remain the same!");
        n = new_n; c = new_c; h = new_h; w = new_w;
    }


    void print(const std::string& name = "Tensor", int max_batch = 1, int max_channel = 1) const {
        std::cout << "\n=== " << name << " [" << n << "x" << c << "x" << h << "x" << w << "] ===\n";

        int batches_to_show = std::min(n, max_batch);
        int channels_to_show = std::min(c, max_channel);

        for (int b = 0; b < batches_to_show; ++b) {
            std::cout << "  [Batch " << b << "]\n";
            for (int ch = 0; ch < channels_to_show; ++ch) {
                if (c > 1) 
                {
                    std::cout << "    Channel " << ch << ":\n";
                }
                for (int y = 0; y < h; ++y) {
                    std::cout << "     ";
                    for (int x = 0; x < w; ++x) {
                        std::cout << std::fixed << std::setprecision(3) << std::setw(8) << (*this)(b, ch, y, x);
                    }
                    std::cout << '\n';
                }
            }
        }
        if (n > max_batch || c > max_channel) {
            std::cout << "     ... (tensor truncated for display)\n";
        }
        std::cout << std::endl;
    }

private:
    [[nodiscard]] size_t index(int bn, int ch, int y, int x) const noexcept {
        // ((bn * c + ch) * h + y) * w + x
        return static_cast<size_t>(((bn * c + ch) * h + y) * w + x);
    }
};

