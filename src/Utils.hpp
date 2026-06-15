// ===========================================================================
// Utils.hpp
// General utility functions for mnist-CNN-from-scratch
// author : @rebwar_ai
// ===========================================================================
#pragma once

#include "Tensor.hpp"
#include <vector>
#include <cassert>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace U {

inline Tensor labels_to_one_hot(const std::vector<int>& labels, int num_classes) {
    int batch = static_cast<int>(labels.size());
    Tensor one_hot(batch, num_classes, 1, 1);
    one_hot.zeros();

    for (int i = 0; i < batch; ++i) {
        int label = labels[i];
        assert(label >= 0 && label < num_classes);
        one_hot(i, label, 0, 0) = 1.0f;
    }
    
    return one_hot;
}


inline int argmax(const Tensor& t, int index, int classes) {
    int base = index * classes;
    float max_val = t.data[base];
    int max_idx = 0;

    for (int i = 1; i < classes; ++i) {
        float v = t.data[base + i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    
    return max_idx;
}


inline std::string format_time_hms(std::chrono::seconds total_seconds)
{
    const auto h = std::chrono::duration_cast<std::chrono::hours>(total_seconds);
    const auto m = std::chrono::duration_cast<std::chrono::minutes>(total_seconds - h);
    const auto s = total_seconds - h - m;

    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << h.count() << " : "
        << std::setw(2) << m.count() << " : "
        << std::setw(2) << s.count();

    return oss.str();
}


void print_ascii(const Tensor& img, const std::string& shades = " .:-=+*#%@")
{
    if (img.n != 1 || img.c != 1) {
        std::cerr << "print_ascii ERROR: tensor must be shape 1x1xHxW\n";
        return;
    }

    int H = img.h;
    int W = img.w;
    int levels = (int)shades.size();

    for (int y = 0; y < H; ++y)
    {
        std::string line;
        line.reserve(W);

        for (int x = 0; x < W; ++x)
        {
            float v = img(0, 0, y, x);
            v = std::clamp(v, 0.0f, 1.0f);

            int idx = (int)(v * (levels - 1));
            line.push_back(shades[idx]);
        }

        std::cout << line << "\n";
    }
}

inline Tensor zeros_like(const Tensor& t) {
    return Tensor(t.n, t.c, t.h, t.w);
}

inline Tensor copy(const Tensor& t) {
    std::vector<float> d = t.data;
    return Tensor(t.n, t.c, t.h, t.w, std::move(d));
}

inline void add_to(Tensor& dst, const Tensor& src) {
    assert(dst.size() == src.size() && "Tensors must have same size for add_to");
    for (size_t i = 0; i < dst.size(); ++i) 
    {
        dst[i] += src[i];
    }
}

} 
