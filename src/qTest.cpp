//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "graph/Derivatives.h"
#include "reinforcement/QAgent.h"
#include "Game.h"

void gameStep(Game &game, int action, double &reward, bool &terminal, int &timeStep){
    game.step(action);
    timeStep++;
    double v = game.value();
    if(v == 0){
        reward = 0.00;
        if(timeStep > 20){
            reward = -1;
            terminal = true;
        }
    }else {
        reward = v;
        terminal = true;
    }
    if(timeStep > 50){
        terminal = true;
    }
}

int main(int argc, char *argv[]) {
    //init
    Operations::init();
    Derivatives::init();
    int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    xt::random::seed(seed);

    Game game;
    game.clipPosition = false;
    game.startXPosition = 1;
    game.startYPosition = 1;
    game.field = {
           {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
           {-1,  0,  0,  0, -1,  0,  0,  0,  0, -1},
           {-1, -1, -1,  0, -1,  0,  0,  1,  0, -1},
           {-1,  2,  0,  0,  0,  0,  0,  0,  0, -1},
           {-1, -1,  0, -1, -1,  0,  0,  0,  0, -1},
           {-1, -1,  0, -1, -1,  0,  0,  0,  0, -1},
           {-1, -1,  0, -1, -1, -1, -1,  2,  0, -1},
           {-1, -1,  0,  0,  0,  0,  0, 10,  0, -1},
           {-1, -1,  1, -1, -1,  0,  0,  0,  0, -1},
           {-1, -1, -1, -1, -1, -1, -1, -1, -1  -1},
    };
    double goal = 6.99;

    QAgent agent(4);
    agent.stepSize = 0.1;
    agent.discountFactor = 0.99;
    agent.upperConfidenceFactor = 10;
    agent.onPolicyTrain = true;

    double avg = 0;
    double evalAvg = 0;
    for(int i = 0; i < 1000000; i++){

        game.resetPosition();
        double r = 0;
        int timeStep = 0;
        bool terminal = false;
        double total = 0;

        bool evaluationEpisode = false;
        if(i % 10 == 0){
            evaluationEpisode = true;
            agent.explore = false;
        }

        while(true){
            Tensor state = {(double)game.xPosition, (double)game.yPosition};

            int action = agent.step(state, r, terminal);
            if(terminal){
                break;
            }
            gameStep(game, action, r, terminal, timeStep);
            total += r;
        }

        if(evaluationEpisode){
            evalAvg = evalAvg * 0.9 + total * 0.1;
            agent.explore = true;
        }else{
            avg = avg * 0.99 + total * 0.01;
        }
        if(i % 100 == 0 && i != 0){
            std::cout << "avg: " << avg << std::endl;
            std::cout << "evalAvg: " << evalAvg << std::endl;
        }

        if(evalAvg > goal || avg > goal){
            std::cout << "avg: " << avg << std::endl;
            std::cout << "evalAvg: " << evalAvg << std::endl;
            std::cout << "goal reached after " << i << " iterations" << std::endl;
            break;
        }

        agent.train(100);
    }


    //play
    {
        agent.explore = false;
        for(int i = 0; i < 3; i++){
            game.resetPosition();
            double r = 0;
            int timeStep = 0;
            bool terminal = false;
            double total = 0;
            while(true){
                std::cout << "\n=========================\n";
                game.print(4);

                Tensor state = {(double)game.xPosition, (double)game.yPosition};
                int action = agent.step(state, r, terminal);
                if(terminal){
                    std::cout << "reward: " << total << std::endl;
                    break;
                }
                gameStep(game, action, r, terminal, timeStep);
                total += r;
            }
        }
    }

    return 0;
}
