//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Node.h"
#include "Operations.h"
#include "Derivatives.h"
#include "toString.h"
#include "Model.h"
#include "Clock.h"
#include "Tensor.h"
#include <iostream>

Node relu(Node x){
    return max(x, x * 0.01);
}

Node sigmoid(Node x){
    return inv(exp(-x) + 1.0);
}

Node tanh(Node x){
    return -(inv(exp(x * 2.0) + 1.0) * 2.0) + 1.0;
}

Node dense(Node x, int in, int out){
    return Node::parameter({out, in}).dot(x) + Node::parameter({out,1});
}

Node network(std::vector<int> layers){
    Node x = Node::input(layers[0]);
    for(int i = 1; i < layers.size(); i++){
        x = dense(x, layers[i-1], layers[i]);
        if(i != layers.size() - 1){
            x = relu(x);
        }
    }
    return x;
}

Node recurrent(std::vector<int> layers){
    Node x = Node::input(layers[0]);
    for(int i = 1; i < layers.size(); i++){
        int in = layers[i-1];
        int out = layers[i];
        x = dense(x, in, out);
        if(i != layers.size() - 1){
            Node buffer = Node::buffer({out, 1});
            x = x + Node::parameter({out, out}).dot(buffer);
            x = relu(x);
            buffer = x * 0.1;
        }
    }
    return x;
}

int main(int argc, char *argv[]){
    //init
    Operations::init();
    Derivatives::init();
    xt::random::seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    //data set
    Tensor input = xt::linspace<double>(0.1, 1, 10);
    Tensor target = input;
    target = xt::vectorize([&](double x){
        return x * x - x * 0.5 + 0.2;
    })(input);
    input = xt::view(input, xt::newaxis(), xt::all());
    target = xt::view(target, xt::newaxis(), xt::all());
    int samples = input.shape(1);


    //model
    Node net = network({1, 10, 10, 10, 1});
    Model model(net);
    model.optimizer = std::make_shared<Adam>(0.001);
    model.optimizer->batchSize = input.shape(1);
    std::cout << toString(model.forward) << std::endl;
    std::cout << toString(model.backward) << std::endl;

    //training
    Clock clock;
    for(int i = 0; i < 10000; i++) {
        double loss = model.samples(input, target, samples);
        if(i % 500 == 0){
            std::cout << "loss: " << loss << std::endl;
        }
    }

    double duration = clock.elapsed();

    //print output values
    std::cout << std::endl;
    for(int c = 0; c < samples; c++){
        Tensor in =  xt::view(input, xt::all(), c, xt::newaxis());
        Tensor out = model.forward.run(in);
        Tensor tar = xt::view(target, xt::all(), c, xt::newaxis());
        std::cout << in(0, 0) << ", " << out(0, 0) << ", " << tar(0, 0) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "time: " << duration << " s" << std::endl;
    return 0;
}
