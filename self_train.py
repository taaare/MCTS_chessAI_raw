# self_train.py
import os
import torch
import traceback
import random
from chess_net import ChessNet
from game_logic_ai import play_game_aivai 
import torch.nn as nn
from tensor_utils import parse_result, train_on_game_data, label_data, save_model

net = ChessNet()

# self_train.py
def self_play_and_train(net, games, training_cycles):
    for cycle in range(training_cycles):
        print(f"Training Cycle {cycle + 1}/{training_cycles}")
        training_data = []
        for game_number in range(games):
            game_data = play_game_aivai(net)
            # Process each game's result and label it
            labeled_data = label_data(game_data, parse_result(game_data[-1][1]))  # Adjust as necessary
            training_data.extend(labeled_data)
        if training_data:
            train_on_game_data(net, training_data, epochs=10)  # Set epochs as needed
        else:
            print("No valid training data available.")
        save_model(net, f'chess_model_cycle_{cycle + 1}.pth')  


