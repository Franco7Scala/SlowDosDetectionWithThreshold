import pandas as pd
import torch
from torch.utils.data import Dataset
from src.support.utils import normalizeValues

class Nidd(Dataset):
    def __init__(self, xy: pd.DataFrame):
        self.xy = xy.drop(["Proto", "sDSb", "dDSb", "Cause", "State", "Label", "Attack Tool"], axis="columns", inplace=True)
        self.xy = normalizeValues(xy)

        self.x = torch.tensor(self.xy.to_numpy()).float()
        self.x = self.x [:, range(0, 44)]
        self.y = torch.tensor(self.xy[['Attack Type']].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)