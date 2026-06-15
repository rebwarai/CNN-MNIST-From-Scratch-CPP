// ===========================================================================
// ImageLoader.hpp
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "Tensor.hpp"
#include "Log.hpp"

#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>


namespace jpeg {

class ImageLoader {
private:
    static constexpr std::array<uint16_t,64> quant_luminance = {{
        16,11,10,16,24,40,51,61,
        12,12,14,19,26,58,60,55,
        14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62,
        18,22,37,56,68,109,103,77,
        24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101,
        72,92,95,98,112,100,103,99
    }};

    static constexpr std::array<int8_t,64> zigzag = {{
        0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
        12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
        35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
        58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
    }};

    void idct_block(int16_t *block) {
        const float PI = 3.14159265358979323846f;
        float tmp[64];
        for (int y = 0; y < 8; ++y) 
        {
            for (int x = 0; x < 8; ++x) 
            {
                float sum = 0.0f;
                for (int v = 0; v < 8; ++v) 
                {
                    for (int u = 0; u < 8; ++u) 
                    {
                        float cu = (u == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                        float cv = (v == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
                        float coeff = static_cast<float>(block[v * 8 + u]);
                        sum += cu * cv * coeff
                             * std::cos(((2*x + 1) * u * PI) / 16.0f)
                             * std::cos(((2*y + 1) * v * PI) / 16.0f);
                    }
                }
                sum *= 0.25f;
                float pix = sum + 128.0f;
                tmp[y * 8 + x] = std::clamp(pix, 0.0f, 255.0f);
            }
        }
        for (int i = 0; i < 64; ++i)
        {
            block[i] = static_cast<int16_t>(std::lround(tmp[i]));
        }
    }
    
    uint32_t bitbuf = 0;
    int bits_avail = 0;
    const std::vector<uint8_t>* raw_data = nullptr;
    size_t byte_pos = 0;
    bool restart_marker_seen = false;

    std::vector<uint8_t> load_raw(const std::string& filepath) {
        
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file) 
        {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        std::streampos sp = file.tellg();
        if (sp < 0) 
        {
            throw std::runtime_error("Failed to determine file size");
        }
        size_t size = static_cast<size_t>(sp);
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(size);
        if (size) {
            file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
            if (!file) {
                throw std::runtime_error("Failed to read entire file");
            }
        }
        
        return buf;
    }

    uint8_t get_next_byte() {
        if (!raw_data) 
        {
            throw std::runtime_error("Raw data not set");
        }
        if (byte_pos >= raw_data->size()) 
        {
            throw std::runtime_error("EOF in bitstream");
        }

        uint8_t b = (*raw_data)[byte_pos++];

        if (b == 0xFF) {
            if (byte_pos < raw_data->size()) {
                uint8_t next = (*raw_data)[byte_pos];
                
                if (next == 0x00) {
                    byte_pos++;
                } else if (next >= 0xD0 && next <= 0xD7) {
                    byte_pos++; 
                    restart_marker_seen = true; 
                    bitbuf = 0;
                    bits_avail = 0;
                    return get_next_byte();
                } 
            } 
        }
        return b;
    }

    bool refill() {
        if (!raw_data)
        {
            return false;
        }
        while (bits_avail <= 16 && byte_pos < raw_data->size()) {
            uint8_t b = get_next_byte();
           
            bitbuf = (bitbuf << 8) | b;
            bits_avail += 8;

        }
        return bits_avail > 0;
    }

    int get_bits(int n) {
        if (n == 0) 
        {
            return 0;
        }
        if (n < 0 || n > 16)
        {
            throw std::invalid_argument("get_bits: n out of range");
        }
        while (bits_avail < n) {
            if (!refill()) 
            {
                throw std::runtime_error("Not enough bits");
            }
        }
        uint32_t mask = ((1u << n) - 1u);
        uint32_t val = (bitbuf >> (bits_avail - n)) & mask;
        bits_avail -= n;

        return static_cast<int>(val);
    }

    inline int extend(int val, int t) {
        if (t <= 0) 
        {
            return 0;
        }
        int vt = 1 << (t - 1);

        if (val < vt) 
        {
            val -= (1 << t) - 1;
        }
        return val;
    }

    struct HuffTable {
        std::array<int, 17> val_index{};
        std::array<int, 17> min_code{};
        std::array<int, 17> max_code{};
        std::vector<int> symbols;

        HuffTable() = default;

        void build_from(const std::array<uint8_t, 17>& bits, const std::vector<uint8_t>& vals) 
        {
            int code = 0;
            int val_offset = 0;
            
            for (int len = 1; len <= 16; ++len) {

                val_index[len] = val_offset;

                if (bits[len] == 0) {
                    min_code[len] = max_code[len] = -1;
                } else {
                    min_code[len] = code;
                    max_code[len] = code + bits[len] - 1;
                    val_offset += bits[len];
                }
                code = (code + bits[len]) << 1;
            }            
            symbols.resize(vals.size());

            for (size_t i = 0; i < vals.size(); ++i)
            {
                symbols[i] = static_cast<int>(vals[i]);
            }
           
        }

        int decode(ImageLoader& loader) const {

            int code = 0; 

            for (int len = 1; len <= 16; ++len) {

                int bit = loader.get_bits(1);
                code = (code << 1) | bit;

                if (min_code[len] != -1 && code >= min_code[len] && code <= max_code[len]) {
                    int idx = val_index[len] + (code - min_code[len]);
                    if (idx < 0 || idx >= static_cast<int>(symbols.size()))
                    {
                        throw std::runtime_error("Huffman index out of range");
                    }
                    
                    return symbols[idx];
                }
            }
            throw std::runtime_error("Huffman decode failed: code too long or corrupt stream");
        }
    };


public:
    Tensor load_jpeg_grayscale(const std::string &filepath) 
    {
        auto raw = load_raw(filepath);

        if (raw.size() < 2 || raw[0] != 0xFF || raw[1] != 0xD8)
        {
            throw std::runtime_error("Invalid JPEG: No SOI");
        }

        size_t pos = 2;
        int width = 0, height = 0;
        bool saw_sos = false;
        uint8_t num_channels = 1; 

        std::map<int, HuffTable> huffman_tables;

        while (pos + 1 < raw.size()) {
            if (raw[pos] != 0xFF) 
            {
                throw std::runtime_error("Expected marker 0xFF");
            }
            uint8_t marker = raw[pos + 1];
            pos += 2;

            if (marker == 0xC4) { 
                uint16_t len = (raw[pos] << 8) | raw[pos + 1];
                pos += 2;
                size_t end_pos = pos + len - 2;
                while (pos < end_pos) {
                    uint8_t ht_info = raw[pos++];

                    int table_class = (ht_info >> 4) & 0x0F; 
                    int table_id = ht_info & 0x0F;

                    std::array<uint8_t,17> bits{};
                    int nvals = 0;
                    for (int i = 1; i <= 16; ++i) {
                        bits[i] = raw[pos++];
                        nvals += bits[i];
                    }

                    std::vector<uint8_t> vals(raw.begin() + pos, raw.begin() + pos + nvals);
                    pos += nvals;

                    HuffTable table;
                    table.build_from(bits, vals);
                    huffman_tables[table_class] = table; 
                }
            }
            else if (marker == 0xC0) { 
                if (pos + 7 > raw.size()) 
                {
                    throw std::runtime_error("Truncated SOF0");
                }
                uint16_t len = (raw[pos] << 8) | raw[pos + 1];
                height = (raw[pos + 3] << 8) | raw[pos + 4];
                width  = (raw[pos + 5] << 8) | raw[pos + 6];
                num_channels = raw[pos + 7]; 

                pos += len;
            }
            else if (marker == 0xDA) { 
                if (pos + 1 >= raw.size()) 
                {
                    throw std::runtime_error("Truncated SOS");
                }
                uint16_t len = (raw[pos] << 8) | raw[pos + 1];
                pos += len;
                saw_sos = true;
                break; 
            }
            else { 
                if (pos + 1 >= raw.size()) 
                {
                    break;
                }
                uint16_t len = (raw[pos] << 8) | raw[pos + 1];
                if (len < 2) 
                {
                    throw std::runtime_error("Invalid segment len");
                }
                pos += len;
            }
        }

        if (!saw_sos) 
        {
            throw std::runtime_error("No SOS found");
        }
        if (width == 0 || height == 0) 
        {
            throw std::runtime_error("No SOF0 found");
        }
        if (num_channels != 1 || height != 28 || width != 28) 
        {
            throw std::runtime_error("Only 28x28 grayscale JPEG supported by this loader");
        }
        
        raw_data = &raw;
        byte_pos = pos;
        bitbuf = 0;
        bits_avail = 0;
        int prev_dc = 0;
        
        Tensor img(1,num_channels,28,28);
        std::fill(img.data.begin(), img.data.end(), 0.0f);

        int mcu_cols = (width  + (8-1)) / 8;   
        int mcu_rows = (height + (8-1)) / 8;  

        for (int mcu_y = 0; mcu_y < mcu_cols; ++mcu_y) {
            for (int mcu_x = 0; mcu_x < mcu_rows; ++mcu_x) {

                std::array<int16_t,64> coeffs{};
                int dc_size = huffman_tables[0].decode(*this); 
                
                if (restart_marker_seen) {
                    prev_dc = 0;
                    restart_marker_seen = false;
                }
                int dc_diff = (dc_size == 0) ? 0 : extend(get_bits(dc_size), dc_size);
                prev_dc += dc_diff;
                coeffs[0] = static_cast<int16_t>(prev_dc);

                int k = 1;
                while (k < 64) {
                    int ac_sym = huffman_tables[1].decode(*this); 
                    if (ac_sym == 0x00) 
                    {
                        break; 
                    }
                    if (ac_sym == 0xF0) 
                    { 
                        k += 16; continue; 
                    } 
                    int run = (ac_sym >> 4) & 0x0F;
                    int size = ac_sym & 0x0F;
                    k += run;
                    if (k >= 64) 
                    {
                        break;
                    }
                    int ac_val = extend(get_bits(size), size);
                    coeffs[zigzag[k]] = static_cast<int16_t>(ac_val);
                    ++k;
                }

                for (int i = 0; i < 64; ++i) 
                {
                    coeffs[i] *= static_cast<int>(quant_luminance[i]);
                }
                idct_block(coeffs.data());

                for (int y = 0; y < 8; ++y) 
                {
                    for (int x = 0; x < 8; ++x) 
                    {
                        int gy = mcu_y*8 + y;
                        int gx = mcu_x*8 + x;
                        if (gy < 28 && gx < 28)
                        {
                            img(0,0,gy,gx) = std::clamp(static_cast<float>(coeffs[y*8+x])/255.0f, 0.0f, 1.0f);
                        }
                    }
                }
            }
        }

        return img;
    }

};

}
