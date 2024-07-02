This file will document how to run the given code for MCTS project in AI CSE 571. Team code #19

First make sure all required python libraries are installed. There is a requirements.txt included in code folder for the libraries that we have additionally used.

To see the MCTS agent in action, run the following command:

python pacman.py -p MCTS_Agent -l mediumClassic

This is our MCTS agent with default settings and evironment

For testing of other agents, just replace MCTS_Agent with AlphaBetaAgent or Expectimax Agent

FOr the hyperparameters, we have used simDepth, steps, optimism. These all have the best values by default, but can be varied in command as follows:

python pacman.py -p MCTS_Agent -l mediumClassic -a optimism=0.1,simDepth=20,steps=100

make sure that there are no spaces in the arguments specified after -a, as that will throw an error

All the other parameters are the same as defined in the base project, such as -l for map selection, -k for no. of ghosts, -g for type of ghost, and so on

The getScores.py file has code to run the different layouts with all agents, and store the scores in test_results folder

python getScores.py

The memory_check.py file has code to run different layouts and capture memory and time usage, and store in memory_results folder

python memory_check.py

There are two results folders. one is test_results, which gives the scores of the different tests we did with all environment variations, and the other is memory_results, which gives the time and space used for different environments by different agents

The jupyter notebook results.ipynb has all the consolidated results, and the code for the ANOVA test and T-test

The graphs were generated using powerpoint, so there is no code for those. We have included the ppt file with the graphs.
