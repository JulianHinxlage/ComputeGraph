//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Node.h"
#include "Operations.h"
#include "Derivatives.h"
#include "toString.h"
#include "Model.h"
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
    return Node::parameter({out, in}).dot(x) + Node::parameter({out});
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
            Node buffer = Node::buffer({out});
            x = x + Node::parameter({out, out}).dot(buffer);
            x = relu(x);
            buffer = x * 0.1;
        }
    }
    return x;
}

Matrix linearSpace(double a, double b, int count){
    Matrix m;
    m.setZero(1, count);
    for(int i = 0; i < count; i++){
        double val = (double)(i) / (double)(count - 1);
        val = a + val * (b - a);
        m(0, i) = val;
    }
    return m;
}

int main(int argc, char *argv[]){
    Operations::init();
    Derivatives::init();
    Node net = recurrent({1, 10, 10, 10, 1});

    Matrix input = linearSpace(0, 1, 11);
    Matrix target = input;
    target = target.unaryExpr([&](double a){
        return a * a - a * 0.5 + 0.2;
    });

    Model model(net);
    model.optimizer.batchSize = input.cols();
    model.optimizer.learningRate = 0.001;

    std::cout << toString(model.forward) << std::endl;
    std::cout << toString(model.backward) << std::endl;

    for(int i = 0; i < 10000; i++) {
        double loss = model.columnSamples(input, target);
        if(i % 500 == 0){
            std::cout << "loss: " << loss << std::endl;
        }
    }

    std::cout << std::endl;
    for(int c = 0; c < input.cols(); c++){
        Matrix in = input.col(c).matrix();
        Matrix out = model.forward.run(in);
        Matrix tar = target.col(c).matrix();
        std::cout << in << ", " << out << ", " << tar << std::endl;
    }

    return 0;
}
