import torch
import chess
import random
import chess.svg
import chess.pgn
from model import ChessCNN
from traitement import board_to_input_tensor

def coin_flip():
    return 'white' if random.random() < 0.5 else 'black'

def load_model(model_path):
    model = ChessCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_ai_move(model, board):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')

    input_tensor = board_to_input_tensor(board)
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))

    for move, score in zip(legal_moves, outputs.squeeze()): 
        if score.item() > best_score:
            best_score = score.item()
            best_move = move

    return best_move if best_move else legal_moves[0] 

def main():
    print("Tirage au sort pour déterminer qui joue les Blancs")
    human_color = coin_flip()
    print(f"Vous jouez les {'Blancs' if human_color == 'white' else 'Noirs'}.")

    model_path = 'chess_cnn_model.pth'
    model = load_model(model_path)
    board = chess.Board()

    while not board.is_game_over():
        print(board)
        
        if (board.turn == chess.WHITE and human_color == 'white') or (board.turn == chess.BLACK and human_color == 'black'):
            move = None
            while move not in board.legal_moves:
                user_input = input("Votre coup (ex. e2e4) : ")
                try:
                    move = chess.Move.from_uci(user_input)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Coup illégal, veuillez réessayer.")
                    continue
            print(f"Vous jouez : {move}")
        else:
            move = get_ai_move(model, board)
            print(f"IA joue : {move.uci()}")

        board.push(move)

    print("Fin de la partie")
    print(f"Résultat : {board.result()}")

if __name__ == '__main__':
    main()