// ===========================================================================
// Model.hpp
// Sequential model for mnist-cnn-from-scratch
// Supports add(), forward(), backward(), zero_grad(), and Adam updates
// author : @rebwar_ai
// ===========================================================================

#pragma once

#include "Tensor.hpp"
#include "layers/Layer.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"
#include "Loss.hpp"

#include <vector>
#include <memory>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>
#include <sstream>



class Model {

public:
    
    std::vector<std::shared_ptr<Layer>> layers;

    template<typename LayerType, typename... Args>
    void add(Args&&... args) {
        layers.emplace_back(std::make_shared<LayerType>(std::forward<Args>(args)...));
    }

    Tensor forward(const Tensor& x) {
        Tensor out = x;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    void backward(const Tensor& grad) {
        Tensor g = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            g = (*it)->backward(g);
        }
    }

    void zero_grad() {
        for (auto& layer : layers) {
            layer->zero_grad();
        }
    }

    void set_training(bool training) {
        for (auto& layer : layers) {
            layer->set_training(training);
        }
    }

    void step(optim::Adam& optimizer) {
        std::vector<Tensor*> params;
        std::vector<Tensor> grads;

        for (auto& layer : layers) {
            auto layer_params = layer->get_parameters(); 
            auto layer_grads  = layer->get_gradients();  

            params.insert(params.end(), layer_params.begin(), layer_params.end());
            grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
        }

        optimizer.step(params, grads);
    }

    void train(int epochs,
        int batch_size,
        Dataset& train_dataset,
        Dataset& test_dataset,
        float learning_rate = 0.001f,
        float learning_rate_decay = 0.0f)
    {
        using Clock = std::chrono::steady_clock;
        auto train_start = Clock::now();

        loss::CrossEntropyLoss loss_fn;
        optim::Adam optimizer(learning_rate);
        
        optimizer.set_decay(learning_rate_decay,
            optim::Adam::DecayType::Exponential);

        const int steps_per_epoch = train_dataset.size() / batch_size;

        this->set_training(true);

        L::plog(SS << "++++++++++++++++++++++ training info ++++++++++++++++++++++\n");
        L::plog(SS << "training size : "<<train_dataset.size() << "\n");
        L::plog(SS << "test size : "<<test_dataset.size() << "\n");
        L::plog(SS << "learning rate : "<<learning_rate << "\n");
        L::plog(SS << "learning rate decay : "<<learning_rate_decay << "\n");
        L::plog(SS << "batch size : "<<batch_size << "\n");
        L::plog(SS << "-------------- start epochs --------------\n");
        
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int correct = 0;

            L::plog(SS << "epoch : " << epoch <<"/"<<(epochs) << "\n");

            train_dataset.shuffle();

            for (int step = 0; step < steps_per_epoch; ++step) {
                if (step == 0){
                    L::plog(SS << "step: " << step <<"/"<<(steps_per_epoch-1)<< "\n");
                }
                else if(step == (steps_per_epoch-1)){
                    L::plog(SS << "\nstep: " << step <<"/"<<(steps_per_epoch-1)<< "\n");
                }
                else
                {
                    L::plog(SS << "..." << step,true,true);
                    if(step % 12 == 0){
                        L::log("\n");
                    }
                }

                std::vector<Tensor> batch_imgs;
                std::vector<int> batch_labels;
                train_dataset.get_batch(
                    step * batch_size,
                    batch_size,
                    batch_imgs,
                    batch_labels
                );

                Tensor X(batch_imgs.size(), 1, 28, 28);

                for (size_t i = 0; i < batch_imgs.size(); ++i) {
                    std::copy(
                        batch_imgs[i].data.begin(),
                        batch_imgs[i].data.end(),
                        X.data.begin() + i * 28 * 28
                    );
                }

                Tensor logits = this->forward(X);

                Tensor target = U::labels_to_one_hot(batch_labels, 10);
                float batch_loss = loss_fn.forward(logits, target);
                Tensor grad = loss_fn.backward(logits, target);
                
                epoch_loss += batch_loss;

                this->zero_grad();
                this->backward(grad);
                this->step(optimizer);

                for (size_t i = 0; i < batch_labels.size(); ++i) {
                    int pred = U::argmax(logits, static_cast<int>(i), 10);
                    if (pred == batch_labels[i]) correct++;
                }

            }

            L::plog(SS << "claculate LOSS and Accuracy\n");
            float acc = static_cast<float>(correct) /
                        (steps_per_epoch * batch_size);

            L::plog(SS << "Epoch " << epoch
                      << " | Loss: " << (epoch_loss / steps_per_epoch)
                      << " | Accuracy: " << acc * 100.0f << "%\n");

            L::plog(SS << "learning rate : "<< optimizer.current_lr() << "\n");
            
        }
        this->set_training(false);
        auto train_end = Clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);
        L::plog(SS <<"\ntraining time : " << U::format_time_hms(total_sec)<<"\n");
    }

    void evaluate(Dataset& test_dataset){
        L::plog(SS << "evaluating... ");
        this->set_training(false);
        int correct_test = 0;
        const int test_size = test_dataset.size();

        for (int i = 0; i < test_size; ++i) {
            std::vector<Tensor> batch_imgs;
            std::vector<int> batch_labels;
            test_dataset.get_batch(i, 1, batch_imgs, batch_labels);

            Tensor x(1, 1, 28, 28);
            std::copy(
                batch_imgs[0].data.begin(),
                batch_imgs[0].data.end(),
                x.data.begin()
            );

            Tensor logits = this->forward(x);
            int pred = U::argmax(logits, 0, 10);

            if (pred == batch_labels[0])
                correct_test++;
        }

        float test_acc = static_cast<float>(correct_test) / test_size;
        L::plog(SS << "Test accuracy: " << test_acc * 100.0f << "%\n");
    }

    void summary() const {
        L::plog(SS << "Model architecture:\n");
        for (const auto& layer : layers) {
            L::plog(SS << "  - " << layer->name() << "\n");
        }
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving: " + filename);
        }

        const uint32_t magic = 0x4E4E4352;  
        const uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        uint32_t num_layers = layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer : layers) {
            std::string layer_name = layer->name();
            uint32_t name_len = layer_name.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(layer_name.c_str(), name_len);

            layer->save(file);
        }

        file.close();
        std::cout << "Model saved to: " << filename << "\n";
    }
    void load(const std::string& filename) {
        std::cout<<"Loading ...";
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for loading: " + filename);
        }

        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != 0x4E4E4352 || version != 1) {
            throw std::runtime_error("Invalid model file format");
        }

        uint32_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        for (auto& layer : layers) {
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            file.read(&name[0], name_len);

            layer->load(file);
        }

        file.close();
        std::cout << "Model loaded from: " << filename << "\n";
    }
};
