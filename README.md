### Monte Carlo Search Tree Simple NN

This was done as a simple project made from scratch to help me understand pytorch on a foundational basis and uses a Monte Carlo Tree Search to make moves.  The neural network is used to evaluate chess positions and guide the MCTS in choosing the most promising moves during gameplay. This project is designed as a basic framework to demonstrate the integration of MCTS with a machine learning model in a chess engine.

Most recently, I have added the capability for the model to train by playing itself. I need to implement CUDA utilization, or send the compute to the cloud. Currently runs on CPU. 

## TODO

Adjust parameters, run on CUDA cores for training

## Features 

Chess Game Simulation: Play against the AI in a command-line interface.
MCTS Implementation: Uses an MCTS algorithm to explore potential moves.
Neural Network Evaluation: Utilizes a simple neural network to evaluate board states, initially with random weights.


Neural Network architecture: MLP -> input: 768 | hidden: 256 | hidden: 64 | output: 3

Optimizer: Adam

NO TRAINING DATA YET - Monte Carlo Search Probabilities ONLY 
