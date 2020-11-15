//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODEL_H
#define COMPUTEGRAPH_MODEL_H

#include "Sequence.h"
#include "Adam.h"

class Model {
public:
    Adam optimizer;
    Sequence forward;
    Sequence backward;

    Model();
    Model(Node &node);
    void compile(Node &node);
    double sample(const Matrix &input, const Matrix &target);
    double columnSamples(const Matrix &input, const Matrix &target, int epochs = 1);
    double loss(const Matrix &output, const Matrix &target);
    Matrix lossGradient(const Matrix &output, const Matrix &target);
};


#endif //COMPUTEGRAPH_MODEL_H
