import random
import math
import chess
import torch

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def expand(self):
        for move in self.state.legal_moves:
            new_state = self.state.copy()
            new_state.push(move)
            self.children.append(MCTSNode(new_state, self))
        if move not in self.state.legal_moves:
            raise Exception(f"Attempting to make an illegal move: {move}")

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.wins / (child.visits + 1e-6)) + c_param * math.sqrt((2 * math.log(self.visits + 1) / (child.visits + 1e-6)))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def simulate(self, net):
        board_tensor = self.state_to_tensor(self.state)
        with torch.no_grad():
            value = net(board_tensor).item()
        return value

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def state_to_tensor(self, state):
        board_str = state.board_fen().split(" ")[0]
        tensor = torch.zeros(12, 8, 8)
        piece_map = {
            'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5,
            'P': 6, 'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11
        }
        i = 0
        for char in board_str:
            if char.isdigit():
                i += int(char)
            elif char == '/':
                continue
            else:
                row, col = divmod(i, 8)
                tensor[piece_map[char], row, col] = 1
                i += 1
        turn = torch.tensor([1.0 if state.turn == chess.WHITE else 0.0]).unsqueeze(0)
        return torch.cat((tensor.view(1, -1), turn), dim=1)

def mcts(root, net, iterations=1000):
    for _ in range(iterations):
        node = root
        while node.children:
            node = node.best_child()
        if not node.state.is_game_over():
            node.expand()
            if node.children:
                node = random.choice(node.children)
        result = node.simulate(net)
        node.backpropagate(result)
    return root.best_child(c_param=0)
