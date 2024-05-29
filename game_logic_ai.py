import chess
from MCTS import MCTSNode, mcts
from tensor_utils import state_to_tensor, parse_result

def play_game_aivai(net):
    board = chess.Board()
    game_data = []
    while not board.is_game_over():
        print(board)
        root = MCTSNode(board.copy())
        best_move_node = mcts(root, net, iterations=100)
        best_move = best_move_node.state.peek()
        board.push(best_move)
        result = parse_result(board.result())
        board_tensor = state_to_tensor(board)  # Ensure this is a tensor
        game_data.append((board_tensor, result))
    return game_data
