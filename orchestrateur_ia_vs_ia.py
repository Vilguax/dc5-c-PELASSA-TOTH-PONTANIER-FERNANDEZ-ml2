import torch
import chess
import random
import numpy as np
import chess.svg
import chess.pgn
from model import ChessCNN
from traitement import board_to_matrix

def load_model(model_path):
    model = ChessCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_ai_move(model, board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

def play_game(model1, model2):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = get_ai_move(model1, board)
        else:
            move = get_ai_move(model2, board)
        board.push(move)
    return board.result()

def save_game_to_pgn(board, file_path="games.pgn"):
    game = chess.pgn.Game.from_board(board)
    with open(file_path, "a") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)
        
def prepare_training_data(pgn_path):
    games = []
    results = []

    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                matrix = board_to_matrix(board)
                games.append(matrix.reshape(1, 8, 8))
                result = game.headers['Result']
                if result == '1-0':
                    results.append(1)
                elif result == '0-1':
                    results.append(0)
                else:
                    results.append(0.5)

    games_array = np.array(games)
    input_tensors = torch.tensor(games_array, dtype=torch.float32)
    label_tensors = torch.tensor(results, dtype=torch.float32)
    
    return input_tensors, label_tensors


def update_model(model, inputs, labels):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model

def main():
    num_games = 5
    model = load_model('chess_cnn_model.pth')
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True):
            move = get_ai_move(model, board)
            board.push(move)
        save_game_to_pgn(board)

    training_data, labels = prepare_training_data("games.pgn")
    update_model(model, training_data, labels)

    torch.save(model.state_dict(), 'updated_model.pth')

if __name__ == '__main__':
    main()