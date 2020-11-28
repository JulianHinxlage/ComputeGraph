//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "graph/Derivatives.h"
#include "reinforcement/QAgent.h"
#include "reinforcement/PolicyGradientAgent.h"
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
            {-1,  0,  0, -1,  0,  0,  0,  0,  0, -1},
            {-1,  0,  0, -1,  0,  0,  0,  2,  0, -1},
            {-1,  0,  0,  0,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  1,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0,  5,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0, 10,  0, -1},
            {-1, -1,  1, -1, -1,  0,  0,  0,  0, -1},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1  -1},
    };
    double goal = 9.0;


    PolicyGradientAgent agent;
    agent.init(2, 4, {10, 10});
    agent.discountFactor = 0.9;
    agent.policy.optimizer->batchSize = 100;

    MeanBuffer averageReward(100);
    for(int i = 0; i < 10000; i++){

        game.resetPosition();
        double r = 0;
        int timeStep = 0;
        bool terminal = false;
        double total = 0;

        if(i % 100 == 0){
            double var = 0;
            double mean = 0;
            double count = 0;
            agent.policy.forward.eachParameter([&](Tensor &p){
               var += xt::sum(p * p)(0);
               mean += xt::sum(p)(0);
               count += p.size();
            });
            mean /= count;
            var /= count;
            var -= mean * mean;
            std::cout << "parameter mean:      " << mean << std::endl;
            std::cout << "parameter variance:  " << var << std::endl;

            if(std::isnan(mean) || std::isnan(var)){
                std::cout << "error: NaN parameter!" << std::endl;
                return -1;
            }
        }

        while(true){
            Tensor state = {(double)game.xPosition, (double)game.yPosition};

            if(i % 1000 == 0){
                std::cout << "action distribution: " << agent.policy.forward.run(state) << std::endl;
            }

            int action = agent.step(state, r, terminal);
            if(terminal){
                break;
            }
            gameStep(game, action, r, terminal, timeStep);
            total += r;
        }


        averageReward.add(total);
        if(i % 100 == 0 && i != 0){
            std::cout << "average reward: " << averageReward.mean() << std::endl;
        }

        if(averageReward.mean() > goal){
            std::cout << "average reward: " << averageReward.mean() << std::endl;
            std::cout << "goal reached after " << i << " iterations" << std::endl;
            break;
        }

        agent.train(100);
    }


    //play
    {
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
                std::cout << "dist: " << agent.policy.forward.run(state) << std::endl;
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
