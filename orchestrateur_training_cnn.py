import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import chess
import chess.pgn
import io
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from model import ChessCNN

model = ChessCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def result_to_label(result):
    if result == '1-0':
        return 2  # White wins
    elif result == '0-1':
        return 0  # Black wins
    else:
        return 1  # Draw

def moves_to_pgn(moves):
    pgn_string = '[Event "Unknown"]\n'
    pgn_string += '[Site "Unknown"]\n'
    pgn_string += '[Date "????.??.??"]\n'
    pgn_string += '[Round "?"]\n'
    pgn_string += '[White "Unknown"]\n'
    pgn_string += '[Black "Unknown"]\n'
    pgn_string += '[Result "*"]\n\n'
    pgn_string += moves + " *"
    return pgn_string

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

def moves_to_final_position(moves):
    pgn_string = moves_to_pgn(moves)
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    board = game.end().board()
    return board_to_matrix(board)

def process_game(args):
    index, moves = args
    result = moves_to_final_position(moves)
    print(f"Traitement terminé pour la partie à l'indice {index}.")
    return result

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Pour l'entraînement
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Pour la validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')


def main():
    print("Chargement des données")
    data = pd.read_csv('chess_games.csv')
    print("Application des labels")
    data['label'] = data['Result'].apply(result_to_label)

    print("Démarrage du traitement parallèle")
    with Pool(10) as pool:
        results = pool.map(process_game, list(enumerate(data['AN'])))
    print("Traitement parallèle terminé.")

    print("Stockage des positions finales...")
    data['final_position'] = results

    print("Préparation des features pour l'apprentissage")
    features = np.array([pos[np.newaxis, :, :] for pos in data['final_position']])
    labels = data['label'].values

    print("Division des données en train et test sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    print("Création des DataLoaders...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    print("Initialisation du modèle")
    model = ChessCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Démarrage de l'entraînement")
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)
    print("Entraînement terminé.")

    print("Sauvegarde du modèle")
    torch.save(model.state_dict(), 'chess_cnn_model.pth')
    print("Modèle sauvegardé.")

if __name__ == "__main__":
    main()