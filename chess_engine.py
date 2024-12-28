import pandas as pd
import numpy as np
import torch
import time
from torch import optim
from torch.utils.data import DataLoader 
from model import ChessModel
from chess_dataset import ChessDataset
from tqdm import tqdm 



if __name__ == '__main__':

    # check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessModel().to(device)
    dataset = ChessDataset()
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 20
    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0
        model.train()

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            
            optimizer.zero_grad()


            outputs = model(inputs)

            outputs = outputs.squeeze()

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

        # loss of each batch
        print(f'Epoch {epoch + 1}/{epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')
        

    torch.save(model.state_dict(), "nets/value1.pth")