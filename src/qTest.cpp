//
// Copyright (c) 2020 Julian Hinxlage. All rights reserved.
//

#include "graph/Derivatives.h"
#include "reinforcement/QAgent.h"
#include "reinforcement/PolicyGradientAgent.h"
#include "reinforcement/ActorCriticAgent.h"
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
    int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    xt::random::seed(seed);

    Game game;
    game.clipPosition = false;
    game.startXPosition = 1;
    game.startYPosition = 1;
    game.field = {
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {-1,  0,  0, -1,  0,  0,  0,  0,  0, -1},
            {-1,  0,  0,  0,  0,  0,  0,  2,  0, -1},
            {-1,  0,  0,  0,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  1,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0,  0,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0,  5,  0, -1},
            {-1, -1,  0,  0,  0,  0,  0, 10,  0, -1},
            {-1, -1,  1, -1, -1,  0,  0,  0,  0, -1},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1  -1},
    };
    double goal = 9.0;


    ActorCriticAgent agent;
    agent.init(2, 4, {10}, {10, 10}, {10}, 0.05);
    agent.discountFactor = 0.99;
    agent.actorCritic.optimizer->batchSize = 100;

    MeanBuffer averageReward(100);
    for(int i = 0; i < 100000; i++){

        game.resetPosition();
        double r = 0;
        int timeStep = 0;
        bool terminal = false;
        double total = 0;

        while(true){
            Tensor state = {(double)game.xPosition, (double)game.yPosition};

            if(i % 10000 == 0){
                auto &output = agent.actorCritic.forward.runMultiple({state});
                std::cout << "action distribution: " << output[0] << std::endl;
                std::cout << "critic values:       " << output[1] << std::endl;
            }

            int action = agent.step(state, r, terminal);
            if(terminal){
                break;
            }
            gameStep(game, action, r, terminal, timeStep);
            total += r;
        }


        averageReward.add(total);
        if(i % 1000 == 0 && i != 0){
            std::cout << "average reward: " << averageReward.mean() << std::endl;
        }

        if(averageReward.mean() > goal){
            std::cout << "average reward: " << averageReward.mean() << std::endl;
            std::cout << "goal reached after " << i << " iterations" << std::endl;
            break;
        }

        agent.trainAll();
        agent.replayBuffer.clear();
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
                {
                    auto &output = agent.actorCritic.forward.runMultiple({state});
                    std::cout << "action distribution: " << output[0] << std::endl;
                    std::cout << "critic values:       " << output[1] << std::endl;
                }
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
