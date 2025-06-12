import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from src.datasets import cicids
from src.support import utils
from src.support.utils import get_base_dir

paths = [#"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
         f"{get_base_dir()}/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
         #"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         #"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         f"{get_base_dir()}/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
         #"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
         #"C:/Users/black/Desktop/Studio/Università/Tirocinio/Tesi/cicids2017/csvs/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
         ]

print("Processing dateset...")
#-----DataFrame-----#
dataframes = utils.readPaths(paths)
labels = utils.stringLabels(dataframes)
dataframes = utils.removeLabels(dataframes, labels, ["DDoS", "DoS Slowhttptest", "DoS slowloris", "BENIGN"])
dataframes = utils.convertStringsSD(dataframes, labels)
dataframe = pd.concat(dataframes)
ddos_dataframe, slowdos_dataframe = utils.divideDataFrame(dataframe)
slowdos_dataframe[" Label"] = slowdos_dataframe[" Label"].replace(2, 1)
#-----DataFrame-----#

dataset = Cicids.Cicids2017(dataframe, True)

ddos_subset, slowdos_subset = utils.divideDataset(dataset)

dataset.y[dataset.y == 2] = 1
print("Done!")

input_size = dataset.x[ddos_subset.indices].shape[1]

output_size = len(np.unique(dataframe[" Label"]))

batch_size = 256

ddos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(ddos_dataframe[" Label"]), y=ddos_dataframe[" Label"]))
slodos_weights = torch.Tensor(compute_class_weight(class_weight="balanced", classes=np.unique(slowdos_dataframe[" Label"]), y=slowdos_dataframe[" Label"]))

print("Creating DataLoders...")
#-----DataLoaders-----#
x_ddos = dataset.x[ddos_subset.indices]
y_ddos = dataset.y[ddos_subset.indices]
x_slowdos = dataset.x[slowdos_subset.indices]
y_slowdos = dataset.y[slowdos_subset.indices]

x_train_ddos, _, y_train_ddos, _ = utils.splitDataset(x_ddos, y_ddos, 0.7, 0.3)
slowdos_train, slowdos_test = utils.createCustomSplitSlowDos(dataset, y_slowdos, slowdos_subset.indices, 0.7, 125)

ddos_train = TensorDataset(x_train_ddos, y_train_ddos)

slowdos_test_loader = DataLoader(slowdos_test, batch_size=batch_size, shuffle=True)
ddos_train_loader = DataLoader(ddos_train, batch_size=batch_size, shuffle=True)
slowdos_train_loader = DataLoader(slowdos_train, batch_size=batch_size, shuffle=True)

x_train_slowdos, y_train_slowdos = utils.convertDataLoaderToNumpy(slowdos_train_loader)
x_test_slowdos, y_test_slowdos = utils.convertDataLoaderToNumpy(slowdos_test_loader)
#-----Train, Validation and Test DataLoaders-----#
print("Done!")

with (open(f"{get_base_dir()}/pickles_cicids/slowdos_test_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_test_loader, f)

with (open(f"{get_base_dir()}/pickles_cicids/input_size.pkl", "wb")) as f:
    pickle.dump(input_size, f)

with (open(f"{get_base_dir()}/pickles_cicids/output_size.pkl", "wb")) as f:
    pickle.dump(output_size, f)

with (open(f"{get_base_dir()}/pickles_cicids/dos_weights.pkl", "wb")) as f:
    pickle.dump(ddos_weights, f)

with (open(f"{get_base_dir()}/pickles_cicids/slowdos_weights.pkl", "wb")) as f:
    pickle.dump(slodos_weights, f)

with (open(f"{get_base_dir()}/pickles_cicids/dos_train_loader.pkl", "wb")) as f:
    pickle.dump(ddos_train_loader, f)

with (open(f"{get_base_dir()}/pickles_cicids/slowdos_train_loader.pkl", "wb")) as f:
    pickle.dump(slowdos_train_loader, f)

with (open(f"{get_base_dir()}/pickles_cicids/x_train_slowdos.pkl", "wb")) as f:
    pickle.dump(x_train_slowdos, f)

with (open(f"{get_base_dir()}/pickles_cicids/y_train_slowdos.pkl", "wb")) as f:
    pickle.dump(y_train_slowdos, f)

with (open(f"{get_base_dir()}/pickles_cicids/x_test_slowdos.pkl", "wb")) as f:
    pickle.dump(x_test_slowdos, f)

with (open(f"{get_base_dir()}/pickles_cicids/y_test_slowdos.pkl", "wb")) as f:
    pickle.dump(y_test_slowdos, f)