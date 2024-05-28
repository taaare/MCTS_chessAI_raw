import chess
from chess_net import net
from MCTS import MCTSNode, mcts

# Initialize the board
board = chess.Board()

def play_game():
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            print(board.fen())
            #user's move
            user_move = input("Enter your move (e.g., e2e4, Nf3, O-O): ")
            try:
                move = board.parse_san(user_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except ValueError:
                print("Invalid move format. Try again.")
                continue
        else: #computer turn
            print(board.fen())
            root = MCTSNode(board)
            best_move_node = mcts(root, net, iterations=1000)
            best_move = best_move_node.state.peek()
            if best_move not in board.legal_moves:
                print(f"Illegal move detected by MCTS: {best_move}")
                continue
            board.push(best_move)
            #print(f"Computer move: {board.san(best_move)}")

    print("Game over.")
    print(board.result())

if __name__ == "__main__":
    play_game()
