//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_CLOCK_H
#define COMPUTEGRAPH_CLOCK_H

#include <chrono>

class Clock {
public:
    std::chrono::nanoseconds time;

    Clock();
    void reset();
    double elapsed();
};


#endif //COMPUTEGRAPH_CLOCK_H
