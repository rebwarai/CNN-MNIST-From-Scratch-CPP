# 🧠 Build a CNN From Scratch in C++ 98.19 accuracy

> A complete Convolutional Neural Network implemented entirely from scratch in modern C++.

❌ No data augmentation
❌ No TensorFlow
❌ No PyTorch
❌ No OpenCV
❌ No ML libraries
❌ No GPU acceleration (CPU only)

✅ Just C++, mathematics, and a deep understanding of how neural networks actually work.

---

## 🎯 Project Goal

Modern deep learning frameworks make neural networks look simple.

A few lines of code can create a model, train it, and achieve impressive results.

But behind that simplicity lies a tremendous amount of engineering:

- 🧩 Tensor systems
- 🔄 Forward propagation
- ⬅️ Backpropagation
- 🧮 Gradient computation
- 🖼️ Image decoding
- ⚡ Optimizers
- 📊 Training pipelines
- 💾 Model serialization

This project exists to **open the black box**.

Instead of using a framework, we build one.

Every major component is implemented manually in C++ so you can understand exactly what happens behind the scenes.

---
## 🎯 Performance

This project is not intended to be a state-of-the-art benchmark model.

The goal is to understand and implement the core mechanics of deep learning from the ground up.

Despite being built entirely from scratch in C++ and trained on a reduced dataset, the network achieves approximately **98.19% test accuracy**.

### Training Configuration

| Metric | Value |
|----------|----------|
| 🏋️ Training Images | 29,600 |
| 🧪 Test Images | 7,400 |
| 📦 Total Images | 37,000 |
| 🔄 Data Augmentation | None |
| 🖥️ Hardware | CPU |
| 🎯 Test Accuracy | 98.19% |

This result was achieved without:

- ❌ TensorFlow
- ❌ PyTorch
- ❌ OpenCV
- ❌ Machine Learning Frameworks
- ❌ Data Augmentation
- ❌ GPU Training

The accuracy serves as a practical validation that the tensor system, convolution layers, backpropagation, optimizer, and training pipeline are functioning correctly.
---

## ✨ Features

### 🏗️ Core Framework

- 📦 Custom 4D Tensor implementation (NCHW layout)
- 🔗 Sequential model API
- ▶️ Forward propagation
- ⬅️ Backward propagation
- 💾 Model serialization
- 📊 Training & evaluation
- 📝 Logging utilities

### 🧠 Neural Network Layers

- 🔍 Conv2D
- 🔗 Dense (Fully Connected)
- ⚡ ReLU
- 📈 Softmax (Numerically Stable)
- 🏊 Max Pooling
- 🎲 Dropout
- 📄 Flatten

### 🚀 Training Components

- 📉 Cross-Entropy Loss
- ⚙️ Adam Optimizer
- 📐 Gradient Computation
- 🔄 Parameter Updates
- 📊 Training Statistics

### 🖼️ Image Processing

A custom grayscale JPEG decoder built entirely from scratch:

- 🏷️ JPEG Marker Parsing
- 🗜️ Huffman Decoding
- 📉 Quantization Handling
- 🎨 Pixel Reconstruction

No external image libraries are used.

---

## 📊 Dataset

The network is trained on a subset of the MNIST handwritten digit dataset.

| Metric | Value |
|----------|----------|
| 🏋️ Training Images | 29,600 |
| 🧪 Test Images | 7,400 |
| 🔢 Classes | 10 |
| 🖼️ Format | JPEG |
| 🎯 Accuracy | ~98% |

All training is performed entirely on the CPU.

---

## 📁 Project Structure

```text
mnist-CNN-from-scratch/
├── src/
│   ├── Tensor.hpp
│   ├── ImageLoader.hpp
│   ├── Dataset.hpp
│   ├── Loss.hpp
│   ├── layers/
│   │   ├── Activations.hpp
│   │   ├── Conv2D.hpp
│   │   ├── Dense.hpp
│   │   ├── Dropout.hpp
│   │   ├── Flatten.hpp
│   │   ├── Layer.hpp
│   │   └── MaxPool.hpp
│   ├── Optimizer.hpp
│   ├── Log.hpp
│   ├── Utils.hpp
│   └── Model.hpp
├── dataset/
│   ├── trainingSet/
│   └── testSet/
├── main.cpp
├── model000.model
└── log.txt
```

---

## 🎥 YouTube Series

This repository accompanies my YouTube series: [link]

### 🎬 Build a CNN From Scratch in C++

In the series we build and explain every component step-by-step:

- 🧠 Tensor Implementation
- 🔍 Convolution
- ⬅️ Backpropagation
- ⚡ Adam Optimizer
- 📉 Loss Functions
- 🏊 Max Pooling
- 🎲 Dropout
- 📊 Training Pipeline

📺 Playlist:

**[PLACEHOLDER_FOR_PLAYLIST_LINK]**

---

## 🎓 What You'll Learn

By following the code and the video series, you'll learn:

- 🧠 How tensors are stored in memory
- 🔍 How convolution really works
- 📐 How gradients are calculated
- ⬅️ How backpropagation flows through a network
- ⚙️ How Adam optimization updates parameters
- 📊 How complete neural networks are trained from scratch

By the end, you'll understand what happens inside modern deep learning frameworks—not just how to use them.

---

## 👨‍💻 Author

**Rebwar**

📺 YouTube: **@rebwar_ai**

🔗 GitHub: **@rebwarai**

---

## ⭐ Support

If you find this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 📺 Subscribe to the YouTube series
- 🧠 Share it with others interested in AI and C++

---

## 📜 License

MIT License

Copyright © 2026 Rebwar

See the LICENSE file for details.