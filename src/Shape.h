//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_SHAPE_H
#define COMPUTEGRAPH_SHAPE_H

#include <vector>
#include "Operations.h"

class Shape {
public:
    std::vector<int> dimensions;
    Shape();
    Shape(std::initializer_list<int> shape);
    Shape(const std::vector<int> &shape);

    template<typename... T>
    Shape(T... t){
        dimensions = {t...};
    }

    bool operator==(const Shape &shape) const;

    Matrix zeros();
};


#endif //COMPUTEGRAPH_SHAPE_H
