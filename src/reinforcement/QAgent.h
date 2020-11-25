//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_QAGENT_H
#define COMPUTEGRAPH_QAGENT_H

#include "Agent.h"
#include "LookupTable.h"

class QAgent : public Agent{
public:
    int actionCount;
    LookupTable values;
    double explorationRate;
    double stepSize;
    QAgent(int actionCount);
    virtual int policyStep(const Tensor &state) override;
    virtual void train(int steps) override;

    double getQ(const Tensor &state, int action);
    void setQ(const Tensor &state, int action, double q);
    double maxQ(const Tensor &state);
    int argmaxQ(const Tensor &state);
};


#endif //COMPUTEGRAPH_QAGENT_H
