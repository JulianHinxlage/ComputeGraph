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
#include <iostream>

int main(int argc, char *argv[]){
    //init
    Operations::init();
    Derivatives::init();
    xt::random::seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

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
    model.loss = std::make_shared<MeanSquaredError>();
    model.optimizer = std::make_shared<Adam>();
    model.optimizer->batchSize = input.shape(1);
    std::cout << toString(model.forward) << std::endl;
    std::cout << toString(model.backward) << std::endl;

    std::cout << model.totalParameterCount() << " parameters" << std::endl << std::endl;

    //training
    Clock clock;
    for(int i = 0; i < 10000; i++) {
        double loss = model.samples(input, target, samples);
        if(i % 500 == 0){
            std::cout << "loss: " << loss << std::endl;
        }
    }
    model.resetToBest();
    std::cout << "best loss: " << model.bestLoss << std::endl;
    std::cout << "test loss: " << model.loss->value(model.predict(input), target) << std::endl;

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
