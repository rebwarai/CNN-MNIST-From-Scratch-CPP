// ===========================================================================
// Dataset.hpp
// Load MNIST dataset from folders of JPG images
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "Tensor.hpp"
#include "ImageLoader.hpp"
#include "Log.hpp"
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>


namespace fs = std::filesystem;

class Dataset {
public:
    struct Sample {
        std::string path;
        int label;
    };

    std::vector<Sample> samples;

 
    void load(const std::string& root_folder) {
        samples.clear();

        for (int label = 0; label <= 9; ++label) {
            std::string subfolder = root_folder + std::to_string(label);

            if (!fs::exists(subfolder)) {
                throw std::runtime_error("Folder does not exist: " + subfolder);
            }

            for (const auto& entry : fs::directory_iterator(subfolder)) {
                if (entry.path().extension() == ".jpg") {
                    std::string path = entry.path().string();
                    std::replace(path.begin(), path.end(), '\\', '/');

                    samples.push_back({ path, label });
                }
            }
        }

        L::plog(SS << "[INFO] Indexed " << samples.size()
                  << " image paths from " << root_folder << "\n");
    }

    
    void shuffle(unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::shuffle(samples.begin(), samples.end(), gen);
    }

    void get_batch(
        size_t start,
        size_t batch_size,
        std::vector<Tensor>& batch_images,
        std::vector<int>& batch_labels
    ) const {
        batch_images.clear();
        batch_labels.clear();

        jpeg::ImageLoader loader;

        size_t end = std::min(start + batch_size, samples.size());

        for (size_t i = start; i < end; ++i) {
            try {
                Tensor img = loader.load_jpeg_grayscale(samples[i].path);

                for (auto& v : img.data)
                {
                    v /= 255.0f;
                }

                batch_images.push_back(std::move(img));
                batch_labels.push_back(samples[i].label);
            }
            catch (const std::exception& e) {
                std::cerr << "[WARN] Failed to load "
                          << samples[i].path << ": "
                          << e.what() << "\n";
            }
        }
    }

    [[nodiscard]] size_t size() const noexcept {
        return samples.size();
    }

    [[nodiscard]] size_t num_batches(size_t batch_size) const noexcept {
        return (samples.size() + batch_size - 1) / batch_size;
    }
};