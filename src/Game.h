//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#ifndef COMPUTEGRAPH_GAME_H
#define COMPUTEGRAPH_GAME_H

#include "graph/Tensor.h"

#include <stdio.h>
#include <termios.h>
#include <unistd.h>

char getch(){
    int c;
    static struct termios oldt, newt;
    tcgetattr( STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON);
    tcsetattr( STDIN_FILENO, TCSANOW, &newt);
    c = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
    return (char)c;
}

class Game{
public:
    Tensor field;
    int xPosition;
    int yPosition;
    int startXPosition;
    int startYPosition;
    bool clipPosition;

    Game(){
        xPosition = 0;
        yPosition = 0;
        startXPosition = 0;
        startYPosition = 0;
        clipPosition = true;
    }

    void step(int action){
        switch (action) {
            case 0:
                xPosition--;
                break;
            case 1:
                xPosition++;
                break;
            case 2:
                yPosition--;
                break;
            case 3:
                yPosition++;
                break;
        }

        if(clipPosition){
            xPosition = std::min(std::max(0, xPosition), (int)field.shape(1) - 1);
            yPosition = std::min(std::max(0, yPosition), (int)field.shape(0) - 1);
        }else{
            xPosition = (xPosition + field.shape(1)) % field.shape(1);
            yPosition = (yPosition + field.shape(0)) % field.shape(0);
        }
    }

    void resetPosition(){
        xPosition = startXPosition;
        yPosition = startYPosition;
    }

    double value(){
        return field(yPosition, xPosition);
    }

    void print(double player = 2){
        double v = field(yPosition, xPosition);
        field(yPosition, xPosition) = player;
        std::cout << field << std::endl;
        field(yPosition, xPosition) = v;
    }

    void manualPlay(){
        resetPosition();
        while(true) {
            std::cout << "\n============================\n\n";
            print(4);
            std::cout << "value: " << value() << std::endl;

            char c = getch();
            if (c == 'w') {
                step(2);
            }
            if (c == 's') {
                step(3);
            }
            if (c == 'a') {
                step(0);
            }
            if (c == 'd') {
                step(1);
            }
            if(c == 'q'){
                break;
            }
        }
    }

};



#endif //COMPUTEGRAPH_GAME_H
