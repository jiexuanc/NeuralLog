from neurallog.torch_models.transformers import TransformerClassifier
from data_loader import load_darpa
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm


embed_dim = 768
num_heads = 12
ff_dim = 2048
max_len = 3500
dropout = 0.1

if __name__ == "__main__":

    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10

    (x_train, y_train), (x_test, y_test) = load_darpa(npz_file="data-bert.npz")
    x_train = np.array([np.array(lst) for lst in x_train])
    x_test = np.array([np.array(lst) for lst in x_test])
    
    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(embed_dim=embed_dim, ff_dim=ff_dim, max_len=max_len, num_heads=num_heads, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training starting...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Training - Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy * 100:.2f}%')