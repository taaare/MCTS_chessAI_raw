import torch
import chess
import random
import os
import torch.nn as nn
from chess_net import ChessNet

net = ChessNet()

def state_to_tensor(state):
    # Create an empty tensor or a zero tensor to hold the encoded board state
    board_tensor = torch.zeros(12, 8, 8)  # 12 planes for each piece type and color
    piece_map = {
        'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5,
        'P': 6, 'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11
    }

    for i in range(64):
        piece = state.piece_at(i)
        if piece:
            piece_type = piece.symbol()
            row, col = divmod(i, 8)
            board_tensor[piece_map[piece_type], row, col] = 1

    # Flatten the tensor and add the turn layer (1 if white's turn, 0 otherwise)
    board_tensor = board_tensor.flatten()
    turn_tensor = torch.tensor([1.0 if state.turn == chess.WHITE else 0.0])
    full_tensor = torch.cat((board_tensor, turn_tensor))
    return full_tensor.unsqueeze(0)  # Add batch dimension

def train_network(net, training_data, epochs=5):
    if not training_data:
        print("No training data provided.")
        return

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    net.train()

    for epoch in range(epochs):
        random.shuffle(training_data)
        print(f"Training Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for board_tensor, result in training_data:
            optimizer.zero_grad()
            if isinstance(board_tensor, tuple):
                board_tensor = board_tensor[0]  # Ensure board_tensor is a tensor, not a tuple
            prediction = net(board_tensor)
            target = torch.tensor([result], dtype=torch.float32).view(-1, 1)  # Ensure target is the same shape as prediction
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}')


def parse_result(result):
    """Converts game result to numeric format."""
    if result == '1-0':
        return 1  # White wins
    elif result == '0-1':
        return 0  # Black wins
    else:
        return 0.5  # Draw

def train_on_game_data(net, game_data, epochs=5):
    training_data = []
    for board_state, game_result in game_data:
        if isinstance(board_state, chess.Board):
            board_tensor = state_to_tensor(board_state)
        else:
            # Assume board_state is already a tensor
            board_tensor = board_state  # If board_state is already a tensor, use it directly
        training_data.append((board_tensor, game_result))

    train_network(net, training_data, epochs)
    
def label_data(game_data, result):
    # Convert result to a suitable format for training (e.g., 1 for win, 0 for loss)
    result_label = parse_result(result)  # You need to define this function
    return [(data, result_label) for data in game_data]

def save_model(net, path='chess_model.pth'):
    print("Current working directory:", os.getcwd())
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist and if directory is not an empty string
    try:
        torch.save(net.state_dict(), path)
        print(f"Model successfully saved to {path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_model(net, path='chess_model.pth'):
    try:
        net.load_state_dict(torch.load(path))
        net.eval()
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Error loading the model: {e}")