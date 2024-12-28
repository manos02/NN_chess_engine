from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):

    def __init__(self):
        d = np.load('dataset_5M.npz')
        self.X = d['arr_0']
        self.y = d['arr_1']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

