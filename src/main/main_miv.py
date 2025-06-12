import torch
import time
import pickle
import os.path

from src.neural_networks.moe_inspired_vae import MoeInspiredVAE
from src.support import utils
from src.nets.VAENN import VAENN
from src.support.focal_loss import FocalLoss
from src.support.utils import get_base_dir, compute_threshold


utils.seed_everything(1) #seed

input_size = 54 #27 mqtt #44 nidd # 54 cicids
epochs_vae = 50
epochs_miv = 5
n_thresholds = 9
gamma = 32
dataset_name = "cicids"


##############################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}...")

output_size = 2
dim_code = 8
dropout = 0.05
first_weights = pickle.load(open(f"{get_base_dir()}/pickles_{dataset_name}/dos_weights.pkl", 'rb'))
adaptation_weights = pickle.load(open(f"{get_base_dir()}/pickles_{dataset_name}/slowdos_weights.pkl", 'rb'))
first_train_loader = pickle.load(open(f"{get_base_dir()}/pickles_{dataset_name}/dos_train_loader.pkl", 'rb'))
adaptation_train_loader = pickle.load(open(f"{get_base_dir()}/pickles_{dataset_name}/slowdos_train_loader.pkl", 'rb'))
slowdos_test_loader = pickle.load(open(f"{get_base_dir()}/pickles_{dataset_name}/slowdos_test_loader.pkl", 'rb'))
vae_path = f"{get_base_dir()}/vae_model_{gamma}_{dataset_name}_miv.pt"

#-----MIV model NeuralNetwork-----#
vae_model = VAENN(dim_code, input_size, device)
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.0001)

#-----VAE model training-----#
if os.path.isfile(vae_path):
    print("Loading VAE model...")
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))

else:
    print("Training VAE model...")
    vae_model.fit(epochs_vae, vae_optimizer, first_train_loader)
    torch.save(vae_model.state_dict(), vae_path)
    print("VAE model saved!")

#-----MIV model training-----#
thresholds = compute_threshold(vae_model, first_train_loader, n_thresholds)
miv_model = MoeInspiredVAE(vae_model, input_size, thresholds, device)
miv_optimizer = torch.optim.Adam(miv_model.parameters(), lr=0.0001)
miv_criterion = FocalLoss(gamma=gamma, alpha=0.5, reduction="mean")

# freezing VAE
for param in vae_model.parameters():
    param.requires_grad = False

start = time.time()
miv_model.fit(epochs_miv, miv_optimizer, miv_criterion, adaptation_train_loader)
end = time.time()
print(f"Training time base model: {end - start:.2f} seconds")

print(f"Starting MIV testing on train set...")
accuracy, precision, recall, f1, auc, cr, pr_auc = miv_model.evaluate(adaptation_train_loader, miv_criterion, evaluation_on="train")
print("MIV test results:")
print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}, pr_auc: {pr_auc}")
print(cr)

print("-" * 100)

print(f"Starting MIV testing on test set...")
accuracy, precision, recall, f1, auc, cr, pr_auc = miv_model.evaluate(slowdos_test_loader, miv_criterion, evaluation_on="test")
print("MIV test results:")
print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}\npr_auc: {pr_auc}")
print(cr)
