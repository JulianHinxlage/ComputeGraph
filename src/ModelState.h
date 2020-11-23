//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_MODELSTATE_H
#define COMPUTEGRAPH_MODELSTATE_H

#include "Model.h"

class ModelState {
public:
    bool hasValue;
    float value;
    std::vector<Tensor> parameter;

    ModelState();

    void load(Model &model);
    void save(Model &model);

    void saveOnMax(Model &model, double value);
    void saveOnMin(Model &model, double value);
};


#endif //COMPUTEGRAPH_MODELSTATE_H
