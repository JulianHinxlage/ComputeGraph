//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Adam.h"

Adam::Adam(double learningRate, double beta1, double beta2, int batchSize){
    this->learningRate = learningRate;
    this->batchSize = batchSize;
    this->beta1 = beta1;
    this->beta2 = beta2;
    sampleCounter = 0;
    beta1t = 1;
    beta2t = 1;
    parameterDecayRate = 0;
}

static void reshape(Adam::Each each, std::vector<Matrix> &data){
    int i = 0;
    each([&](Matrix &parameter, Matrix &gradient){
        if(data.size() <= i){
            data.push_back(gradient * 0.0);
        }else{
            if(gradient.rows() != data[i].rows() || gradient.cols() != data[i].cols()){
                data[i] = gradient * 0.0;
            }
        }
        i++;
    });
}

void Adam::update(Each each, int samples){
    sampleCounter += samples;
    if(sampleCounter >= batchSize){
        each([&](Matrix &p, Matrix &g){
            g /= sampleCounter;
        });
        updateRule(each);
        sampleCounter = 0;
    }
}

void Adam::updateRule(Each each){
    reshape(each, gradientMomentum1);
    reshape(each, gradientMomentum2);

    beta1t *= beta1;
    beta2t *= beta2;
    int i = 0;
    each([&](Matrix &parameter, Matrix &gradient){
        gradientMomentum1[i] = gradientMomentum1[i] * beta1 + gradient * (1.0 - beta1);
        gradientMomentum2[i] = gradientMomentum2[i] * beta2 + gradient.cwiseProduct(gradient) * (1.0 - beta2);

        parameter -= gradientMomentum1[i].binaryExpr(gradientMomentum2[i], [&](double m, double v){
            m /= (1.0 - beta1t);
            v /= (1.0 - beta2t);
            return learningRate * m / (sqrt(v) + 1e-10);
        }) + learningRate * parameterDecayRate * parameter;

        gradient *= 0;
        i++;
    });
}
