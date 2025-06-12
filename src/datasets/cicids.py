from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from src.support.utils import removeCollinearFeatures, normalizeValues

class Cicids2017(Dataset):
    def __init__(self, xy: pd.DataFrame, preprocessData: Optional[bool] = False):
        if preprocessData:
            self.xy = xy.drop([' Destination Port'], axis="columns", inplace=True)
            self.xy = normalizeValues(xy)
            self.xy = removeCollinearFeatures(xy, 0.95)
        else:
            self.xy = xy

        self.x = torch.tensor(self.xy.to_numpy()).float()
        self.x = self.x[:, range(0, 54)]
        self.y = torch.tensor(self.xy[[' Label']].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)
