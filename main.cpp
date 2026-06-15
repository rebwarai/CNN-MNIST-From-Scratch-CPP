// ===========================================================================
// main.cpp
// author : @rebwar_ai
// ===========================================================================

#include "src/Dataset.hpp"
#include "src/layers/Activations.hpp"
#include "src/Optimizer.hpp"
#include "src/Model.hpp"
#include "src/layers/Conv2D.hpp"
#include "src/layers/MaxPool.hpp"
#include "src/layers/Flatten.hpp"
#include "src/layers/Dense.hpp"
#include "src/layers/Dropout.hpp"

#include <iostream>



using namespace std;


int main() {
    try { 
        // ------------------------------------------------------------
        // Load dataset
        // ------------------------------------------------------------
        Dataset train_dataset, test_dataset;
        train_dataset.load("dataset/trainingSet/");
        test_dataset.load("dataset/testSet/");
        train_dataset.shuffle();

        // ------------------------------------------------------------
        // Define model
        // ------------------------------------------------------------
        Model model;
        model.add<Conv2D>(1, 32, 5, 1, 0);      // 28x28 → 24x24
        model.add<activations::ReLU>();
        model.add<MaxPool>(2, 2);               // 24x24 → 12x12
        model.add<Conv2D>(32, 64, 5, 1, 0);     // 12x12 → 8x8
        model.add<activations::ReLU>();
        model.add<MaxPool>(2, 2);               // 8x8 → 4x4
        model.add<Flatten>();
        model.add<Dense>(64 * 4 * 4, 256);
        model.add<activations::ReLU>();
        model.add<Dropout>(0.25f);              // ← 25% dropout Dropout 
        model.add<Dense>(256, 10);               // logits
        
        model.summary();

        string train_model;
        string model_name = "model000.model";
        cout << "Do you want to train a new model ? (y/n) :";
        cin >> train_model;

        if (train_model == "y" || train_model == "Y")
        {
            // ------------------------------------------------------------
            // Training settings 
            // ------------------------------------------------------------
            const int epochs = 12;
            const int batch_size = 64;
            const float learning_rate = 0.001f;
            const float learning_rate_decay = 5e-5f;
            // ------------------------------------------------------------
            // Training loop
            // ------------------------------------------------------------
            model.train(epochs,batch_size,train_dataset,
                        test_dataset,learning_rate,
                        learning_rate_decay);
            // ------------------------------------------------------------
            // Test evaluation
            // ------------------------------------------------------------

            model.save(model_name);
        }
        else
        {
            model.load(model_name);
        }
        
        // model.evaluate(test_dataset);

        model.set_training(false);
        std::string filepath = "dataset/testSet/3/img_6702.jpg";
        jpeg::ImageLoader loader;
        Tensor img = loader.load_jpeg_grayscale(filepath);
        Tensor logits = model.forward(img);
        
        int pred = U::argmax(logits, 0, 10);
        cout << "test image : " << pred;

    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception 1: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "[ERROR] Unknown exception occurred\n";
        return 1;
    }

    return 0;
}
