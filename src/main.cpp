//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Node.h"
#include "Sequence.h"
#include "Operations.h"
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

    Node net = network({1, 3, 3, 3, 1});

    Sequence s(net);
    std::cout << s.toString() << std::endl;
    s.eachParameter([](Matrix &parameter){
        parameter.setRandom();
    });

    Matrix in;
    in.setConstant(1, 1, 1.0f);

    Matrix out = s.run(in);
    std::cout << out << std::endl;
    return 0;
}
