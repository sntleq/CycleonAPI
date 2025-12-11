"""
SageMaker Training Script for LSTM Time Series Prediction
This trains a deep learning model to predict next occurrence times and probabilities
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class ItemLSTM(nn.Module):
    """LSTM model for predicting next delta times"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(ItemLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return self.relu(prediction)  # Ensure positive predictions


def prepare_sequences(df, seq_length=5):
    """
    Convert delta_minutes into sequences for LSTM
    Each sequence of seq_length deltas predicts the next delta
    """
    sequences = []
    targets = []

    # Group by item and shop
    for (item, shop), group in df.groupby(['item', 'shop']):
        group = group.sort_values('timestamp').reset_index(drop=True)
        deltas = group['delta_minutes'].values

        # Create sequences
        for i in range(len(deltas) - seq_length):
            seq = deltas[i:i + seq_length]
            target = deltas[i + seq_length]
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


def train_model(train_loader, model, criterion, optimizer, device, epochs=50):
    """Training loop"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device).unsqueeze(-1)  # Add feature dimension
            targets = targets.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')


def main():
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--seq-length', type=int, default=5)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    print(f'Loading data from {args.train}')
    df = pd.read_csv(f'{args.train}/items.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f'Data shape: {df.shape}')
    print(f'Unique items: {df["item"].nunique()}')

    # Prepare sequences
    print('Preparing sequences...')
    X, y = prepare_sequences(df, seq_length=args.seq_length)
    print(f'Created {len(X)} sequences')

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(X_scaled, y)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = ItemLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    print('Starting training...')
    train_model(train_loader, model, criterion, optimizer, device, epochs=args.epochs)

    # Save model and scaler
    print(f'Saving model to {args.model_dir}')
    torch.save(model.state_dict(), f'{args.model_dir}/lstm_model.pth')
    joblib.dump(scaler, f'{args.model_dir}/scaler.pkl')

    # Save model config for inference
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'seq_length': args.seq_length
    }
    joblib.dump(config, f'{args.model_dir}/config.pkl')

    print('Training complete!')


if __name__ == '__main__':
    main()