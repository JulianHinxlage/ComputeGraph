//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Loss.h"

double MeanSquaredError::value(const Tensor &output, const Tensor &target) {
    return xt::sum((output - target) * (output - target))() / output.shape(1) / 2;
}

Tensor MeanSquaredError::gradient(const Tensor &output, const Tensor &target) {
    return (output - target) / output.shape(1);
}

double CrossEntropy::value(const Tensor &output, const Tensor &target) {
    return -xt::sum(target * xt::log(output))() / output.shape(1);
}

Tensor CrossEntropy::gradient(const Tensor &output, const Tensor &target) {
    return -(target / output) / output.shape(1);
}

double BinaryCrossEntropy::value(const Tensor &output, const Tensor &target) {
    return -xt::sum(target * xt::log2(output) + (1 - target) * xt::log2(1 - output))() / output.shape(1);
}

Tensor BinaryCrossEntropy::gradient(const Tensor &output, const Tensor &target) {
    return -(target / output + (1 - target) / (output - 1)) / output.shape(1);
}
