import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

from src.datasets.nidd import Nidd
from src.support import utils
from src.support.utils import get_base_dir

path = [f"{get_base_dir()}/csvs/nidd/Combined.csv"]

batch_size = 256

print("Processing dataset...")
#-----DataFrame-----#
dataframes = utils.readPaths(path)
labels = utils.stringLabels(dataframes)
dataframes = utils.removeLabels(dataframes, labels, ["Benign", "HTTPFlood", "SlowrateDoS"])
dataframes = utils.convertStringsSD(dataframes, labels)
dataframe = pd.concat(dataframes)
dos_dataframe, slowdos_dataframe = utils.divideDataFrame(dataframe)
slowdos_dataframe["Attack Type"] = slowdos_dataframe["Attack Type"].replace(2, 1)
#-----DataFrame-----#

dataset = Nidd(dataframe)

dos_subset, slowdos_subset = utils.divideDataset(dataset)

dos_count = np.unique(dataset.y[dos_subset.indices], return_counts=True)
slow_count = np.unique(dataset.y[slowdos_subset.indices], return_counts=True)

dataset.y[dataset.y == 2] = 1
print("Done!")

print("Creating DataLoaders...")
#-----Dataloaders-----#
x_dos = dataset.x[dos_subset.indices]
y_dos = dataset.y[dos_subset.indices]
x_slowdos = dataset.x[slowdos_subset.indices]
y_slowdos = dataset.y[slowdos_subset.indices]

x_train_dos, _, y_train_dos, _ = utils.splitDataset(x_dos, y_dos, 0.7, 0.3)
slowdos_train, slowdos_test = utils.createCustomSplitSlowDos(dataset, y_slowdos, slowdos_subset.indices, 0.7, 5)

dos_train = TensorDataset(x_train_dos, y_train_dos)

slowdos_test_loader = DataLoader(slowdos_test, batch_size=batch_size, shuffle=True)
dos_train_loader = DataLoader(dos_train, batch_size=batch_size, shuffle=True)
slowdos_train_loader = DataLoader(slowdos_train, batch_size=batch_size, shuffle=True)

x_train_slowdos, y_train_slowdos = utils.convertDataLoaderToNumpy(slowdos_train_loader)
x_test_slowdos, y_test_slowdos = utils.convertDataLoaderToNumpy(slowdos_test_loader)
#-----Dataloaders-----#
print("Done!")

input_size = dataset.x[dos_subset.indices].shape[1]
output_size = 2

dos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(dos_dataframe["Attack Type"]), y=dos_dataframe["Attack Type"]))
slodos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(slowdos_dataframe["Attack Type"]), y=slowdos_dataframe["Attack Type"]))

with (open(f"{get_base_dir()}/pickles/slowdos_test_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_test_loader, f)

with (open(f"{get_base_dir()}/pickles/input_size.pkl", "wb")) as f:
    pickle.dump(input_size, f)

with (open(f"{get_base_dir()}/pickles/output_size.pkl", "wb")) as f:
    pickle.dump(output_size, f)

with (open(f"{get_base_dir()}/pickles/dos_weights.pkl", "wb")) as f:
    pickle.dump(dos_weights, f)

with (open(f"{get_base_dir()}/pickles/slowdos_weights.pkl", "wb")) as f:
    pickle.dump(slodos_weights, f)

with (open(f"{get_base_dir()}/pickles/dos_train_loader.pkl", "wb")) as f:
    pickle.dump(dos_train_loader, f)

with (open(f"{get_base_dir()}/pickles/slowdos_train_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_train_loader, f)

with (open(f"{get_base_dir()}/pickles/x_train_slowdos.pkl", "wb")) as f:
    pickle.dump(x_train_slowdos, f)

with (open(f"{get_base_dir()}/pickles/y_train_slowdos.pkl", "wb")) as f:
    pickle.dump(y_train_slowdos, f)

with (open(f"{get_base_dir()}/pickles/x_test_slowdos.pkl", "wb")) as f:
    pickle.dump(x_test_slowdos, f)

with (open(f"{get_base_dir()}/pickles/y_test_slowdos.pkl", "wb")) as f:
    pickle.dump(y_test_slowdos, f)