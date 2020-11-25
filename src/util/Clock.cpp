//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "Clock.h"


std::chrono::nanoseconds now(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
}

Clock::Clock() {
    reset();
}

void Clock::reset() {
    time = now();
}

double Clock::elapsed() {
    return (double)(unsigned long long)(now().count() - time.count()) / 1000.0 / 1000.0 / 1000.0;
}
