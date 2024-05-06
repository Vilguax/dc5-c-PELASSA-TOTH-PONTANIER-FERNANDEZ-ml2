import pandas as pd
import numpy as np
import chess
import chess.pgn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import io
from model import ChessCNN

data = pd.read_csv('chess_games.csv')

def result_to_label(result):
    if result == '1-0':
        return 2
    elif result == '0-1':
        return 0
    else:
        return 1

def moves_to_final_position(moves):
    game = chess.pgn.read_game(io.StringIO(moves))
    board = game.end().board()
    return board_to_matrix(board)

def board_to_matrix(board):
    pieces = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
              'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}
    matrix = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            if piece:
                matrix[i, j] = pieces[str(piece)]
    return matrix

data['label'] = data['Result'].apply(result_to_label)
data['final_position'] = data['AN'].apply(moves_to_final_position)

data['WhiteElo'] = (data['WhiteElo'] - data['WhiteElo'].mean()) / data['WhiteElo'].std()
data['BlackElo'] = (data['BlackElo'] - data['BlackElo'].mean()) / data['BlackElo'].std()

features = np.array(list(data['final_position']))
labels = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ChessCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}: Training Loss: {running_loss / len(train_loader)}')

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Epoch {epoch+1}: Validation Loss: {val_loss / len(test_loader)}, Accuracy: {100 * correct / total}%')

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)