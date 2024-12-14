import pandas as pd
import numpy as np
import torch
import time
from torch import optim
from torch.utils.data import DataLoader 
from model import ChessModel
from chess_dataset import ChessDataset
from tqdm import tqdm 
from generate_training_set import get_dataset



if __name__ == '__main__':


    X, y = get_dataset(5000)

    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ChessModel(num_classes=num_classes).to(device)
    
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50
    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        model.train()

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels) # calculate loss
            loss.backward() # backward pass

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(f'Epoch {epoch + 1 + 50}/{epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')
        


