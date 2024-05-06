import numpy as np
import torch
import chess

def fen_to_matrix(fen):
    pieces = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
              'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}
    empty = 0
    board = np.zeros((8, 8), dtype=int)
    fen = fen.split()[0]
    rows = fen.split('/')
    for i, row in enumerate(rows):
        col_index = 0
        for char in row:
            if char.isdigit():
                col_index += int(char)
            else:
                board[i, col_index] = pieces[char]
                col_index += 1
    return board


def board_to_input_tensor(board):
    piece_to_value = {
        'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
        'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        None: 0
    }
    matrix = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            matrix[i, j] = piece_to_value[str(piece)] if piece else 0

    input_tensor = torch.tensor(matrix, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

def board_to_matrix(board):
    piece_to_value = {
        'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
        'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        None: 0  # pour une case vide
    }
    matrix = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            matrix[i, j] = piece_to_value[str(piece)] if piece else 0
    return matrix