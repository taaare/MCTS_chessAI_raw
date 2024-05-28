### Monte Carlo Search Tree Simple NN

This was done as a simple project made from scratch to help me understand pytorch on a foundational basis and uses a Monte Carlo Search Tree to make moves.  The neural network is used to evaluate chess positions and guide the MCTS in choosing the most promising moves during gameplay. This project is designed as a basic framework to demonstrate the integration of MCTS with a machine learning model in a chess engine.

## Features 

Chess Game Simulation: Play against the AI in a command-line interface.
MCTS Implementation: Uses an MCTS algorithm to explore potential moves.
Neural Network Evaluation: Utilizes a simple neural network to evaluate board states, initially with random weights.

Neural Network architecture: MLP -> input: 768 | hidden: 256 | hidden: 64 | output: 3
Optimizer: Adam
NO TRAINING DATA - Monte Carlo Search Probabilities ONLY 
