//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Node.h"
#include "Sequence.h"
#include "Operations.h"
#include "Differentiator.h"
#include "Derivatives.h"
#include "toString.h"
#include <iostream>

Node relu(Node x){
    return max(x, x * 0.01);
}

Node dense(Node x, int in, int out){
    return Node::parameter({out, in}) * x + Node::parameter({out});
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

int main(int argc, char *argv[]){
    Operations::init();
    Derivatives::init();

    Node net = network({1, 10, 10, 10, 1});

    Sequence forward;
    forward.generate(net);
    std::cout << toString(forward) << std::endl;
    forward.eachParameter([](Matrix &parameter){
        parameter.setRandom();
    });

    Differentiator diff;
    std::vector<Node> gradients;
    Sequence backward;
    backward.setParent(forward);
    backward.generate(diff.differentiate(net, gradients));
    for(auto &n : gradients){
        backward.generate(n);
    }
    std::cout << toString(backward) << std::endl;

    Matrix in;
    in.setConstant(1, 1, 1.0f);

    Matrix target;
    target.setConstant(1, 1, 0.42f);

    Matrix out;
    in.setConstant(1, 1, 0.0f);

    for(int i = 0; i < 100; i++){
        out = forward.run(in);
        if(i % 50 == 0){
            std::cout << out << std::endl;
        }
        backward.run(out - target);
        backward.eachGradient([](Matrix &parameter, Matrix &gradient){
           parameter -= gradient * 0.01;
           gradient *= 0.0f;
        });
    }
    return 0;
}
