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

static void reshape(Adam::Each each, std::vector<Tensor> &data){
    int i = 0;
    each([&](Tensor &parameter, Tensor &gradient){
        if(data.size() <= i){
            data.push_back(gradient * 0.0);
        }else{
            if(data[i].shape() != gradient.shape()){
                data[i] = gradient * 0.0;
            }
        }
        i++;
    });
}

void Adam::update(Each each, int samples){
    sampleCounter += samples;
    if(sampleCounter >= batchSize){
        each([&](Tensor &p, Tensor &g){
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
    each([&](Tensor &parameter, Tensor &gradient){
        gradientMomentum1[i] = gradientMomentum1[i] * beta1 + gradient * (1.0 - beta1);
        gradientMomentum2[i] = gradientMomentum2[i] * beta2 + (gradient * gradient) * (1.0 - beta2);

        parameter -= xt::vectorize([&](double m, double v){
            m /= (1.0 - beta1t);
            v /= (1.0 - beta2t);
            return learningRate * m / (std::sqrt(v));
        })(gradientMomentum1[i], gradientMomentum2[i]) + learningRate * parameterDecayRate * parameter;

        gradient *= 0;
        i++;
    });
}
