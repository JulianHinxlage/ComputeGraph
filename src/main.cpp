//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Operations.h"
#include "Derivatives.h"
#include "toString.h"
#include "Model.h"
#include "Clock.h"
#include "Tensor.h"
#include "ModelBuilder.h"
#include "ModelState.h"
#include <iostream>
#include <fstream>
#include "xtensor/xcsv.hpp"

void test(){
    ModelBuilder encoderDef;
    encoderDef.input(100);
    encoderDef.dense(20);
    encoderDef.relu();
    encoderDef.dense(10);
    encoderDef.tanh();
    encoderDef.dense(10);
    encoderDef.relu();


    ModelBuilder decoderDef;
    decoderDef.input(10);
    decoderDef.dense(10);
    decoderDef.relu();
    decoderDef.dense(20);
    decoderDef.tanh();
    decoderDef.dense(100);
    decoderDef.relu();

    Model encoder;
    encoder.compile(encoderDef.node);
    encoder.optimizer = std::make_shared<Adam>();

    Model decoder;
    decoder.compile(decoderDef.node);
    decoder.optimizer = std::make_shared<Adam>();


    Tensor data;
    data = xt::zeros<double>({100, 90});
    for(int i = 0; i < 90; i++){
        for(int j = 0; j <= 10; j++){
            data(i + j, i) = 1.0;
        }
    }


    //load
    {
        int index = 0;
        encoder.forward.eachParameter([&](Tensor &p){
            std::ifstream file("encoder" + std::to_string(index) + ".csv");
            if(file.is_open()) {
                p = xt::load_csv<double>(file);
            }
            index++;
        });
        decoder.forward.eachParameter([&](Tensor &p){
            std::ifstream file("encoder" + std::to_string(index) + ".csv");
            if(file.is_open()) {
                p = xt::load_csv<double>(file);
            }
            index++;
        });
    }

    int epochs = 10000;
    int samples = 90;

    ModelState encoderState;
    ModelState decoderState;

    //train
    MeanSquaredError lossFunction;
    for(int epoch = 0; epoch <= epochs; epoch++){

        Tensor latent = encoder.forward.run(data);
        Tensor reconstruction = decoder.forward.run(latent);

        Tensor lossGradient = lossFunction.gradient(reconstruction, data);
        double loss = lossFunction.value(reconstruction, data);

        encoderState.saveOnMin(encoder, loss);
        decoderState.saveOnMin(decoder, loss);

        if(epoch % 100 == 0){
            std::cout << "epoch: " << epoch << "/" << epochs << "  loss: " << loss << std::endl;
        }

        Tensor latentGradient = decoder.backward.run(lossGradient);
        encoder.backward.run(latentGradient);

        encoder.optimizer->update([&](auto &c){encoder.backward.eachGradient(c);}, samples);
        decoder.optimizer->update([&](auto &c){decoder.backward.eachGradient(c);}, samples);
    }

    encoderState.load(encoder);
    decoderState.load(decoder);

    Tensor latent = encoder.forward.run(data);
    Tensor reconstruction = decoder.forward.run(latent);

    xt::print_options::set_edge_items(100000);
    xt::print_options::set_line_width(100000);
    std::cout << latent << std::endl;
    std::cout << "\n\n";
    std::cout << xt::cast<double>(reconstruction > 0.5) << std::endl;


    //save
    {
        int index = 0;
        encoder.forward.eachParameter([&](Tensor &p){
            std::ofstream file("encoder" + std::to_string(index) + ".csv");
            xt::dump_csv(file, p);
            index++;
        });
        decoder.forward.eachParameter([&](Tensor &p){
            std::ofstream file("encoder" + std::to_string(index) + ".csv");
            xt::dump_csv(file, p);
            index++;
        });
    }

}

int main(int argc, char *argv[]){
    //init
    Operations::init();
    Derivatives::init();
    xt::random::seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    //test();
    //return 0;

    //data set
    Tensor input = xt::linspace<double>(0.0, 1, 11);
    input = xt::view(input, xt::newaxis(), xt::all());
    Tensor target = xt::vectorize([&](double x){
        return x * x - x * 0.5 + 0.2;
    })(input);
    int samples = input.shape(1);


    //model
    ModelBuilder net;
    net.input(1);
    net.dense(10);
    net.relu();

    net.residual(1);
    net.residual();
    net.dense(10);
    net.relu();
    net.residual();
    net.residual();
    net.dense(10);
    net.relu();
    net.residual();
    net.residual(1);

    net.dropout(0.25);
    net.dense(1);

    Model model(net.node);
    model.lossFunction = std::make_shared<MeanSquaredError>();
    model.optimizer = std::make_shared<Adam>();
    model.optimizer->batchSize = input.shape(1);
    std::cout << toString(model.forward) << std::endl;
    std::cout << toString(model.backward) << std::endl;

    std::cout << model.totalParameterCount() << " parameter" << std::endl << std::endl;

    ModelState state;

    //training
    Clock clock;
    for(int i = 0; i < 10000; i++) {
        double loss = model.fit(input, target, samples);

        state.saveOnMin(model, loss);

        if(i % 500 == 0){
            std::cout << "loss: " << loss << std::endl;
        }
    }

    state.load(model);

    double duration = clock.elapsed();

    //print output values
    std::cout << std::endl;
    for(int c = 0; c < samples; c++){
        Tensor in =  xt::view(input, xt::all(), c, xt::newaxis());
        Tensor out = model.predict(in);
        Tensor tar = xt::view(target, xt::all(), c, xt::newaxis());
        std::cout << xt::view(in, xt::all(), 0) << ", " << xt::view(out, xt::all(), 0) << ", " << xt::view(tar, xt::all(), 0) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "time: " << duration << " s" << std::endl;
    return 0;
}
