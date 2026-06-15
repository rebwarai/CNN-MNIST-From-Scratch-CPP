// ===========================================================================
// ImageLoader_Test.cpp
// Example usage of the custom JPEG ImageLoader.hpp with ASCII visualization.
// NOTE: You must provide a valid 28x28 grayscale JPEG file (like an MNIST example).
// author : @rebwar_ai
// ===========================================================================

#include "../src/Tensor.hpp"
#include "../src/ImageLoader.hpp"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>

void print_ascii(const Tensor& img, const std::string& shades = " .:-=+*#%@")
{
    // Must be 1×1×H×W tensor
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

// tensor: [1x1x28x28]
void save_tensor_as_pgm(const Tensor& tensor, const std::string& filename) {
    // eog output.pgm -> to open image in WSL
    int height = 28;
    int width = 28;

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] Cannot open file for writing: " << filename << "\n";
        return;
    }

    // Write PGM header
    out << "P5\n" << width << " " << height << "\n255\n";

    // Tensor values are [0,1], convert to 0-255
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = tensor(0,0,y,x);  // batch=0, channel=0
            uint8_t pixel = static_cast<uint8_t>(std::clamp(val * 255.0f, 0.0f, 255.0f));
            out.write(reinterpret_cast<char*>(&pixel), 1);
        }
    }

    out.close();
    std::cout << "[INFO] Saved tensor as PGM: " << filename << "\n";
}
// --- Main Execution ---

int main(int argc, char* argv[]) {


    std::string filepath = "../dataset/testSet/2/img_4495.jpg";
    jpeg::ImageLoader loader;

    try {
        Tensor img = loader.load_jpeg_grayscale(filepath);
        img.print("Test Image", 1, 1);
        print_ascii(img);
        // save_tensor_as_pgm(img, "output.pgm");
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Failed to load JPEG: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}

