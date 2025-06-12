import copy
import torch
import numpy
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader
from src.support.utils import get_base_dir


class MoeInspiredVAE(nn.Module):

    def __init__(self, vae, input_size, thresholds, device):
        super(MoeInspiredVAE, self).__init__()
        self.device = device
        self.vae = vae
        self.thresholds = thresholds
        self.selector = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(thresholds)),
        )
        self.to(self.device)

    def forward(self, x):
        error = F.mse_loss(self.vae(x)[2], x, reduction="none")
        error = torch.mean(error, dim=1)
        vae_outputs = torch.stack([error > self.thresholds[i] for i in range(len(self.thresholds))], dim=1)
        selector_outputs = self.selector(x)
        selector_outputs = F.softmax(selector_outputs, dim=1)
        output = vae_outputs * selector_outputs
        output = torch.sum(output, dim=1)
        complement = 1.0 - output
        return torch.stack([complement, output], dim=1)


# -----train and test-----#
    def _train_epoch(self, train_loader, optimizer, criterion):
        self.train()
        loss_sum = 0
        count = 0
        for i, content in enumerate(train_loader):
            optimizer.zero_grad()
            target = content[1].to(self.device).view(-1).long()
            input = content[0].to(self.device)
            output = self(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            count += 1

        return loss_sum / count

    def evaluate(self, loader, criterion, evaluation_on="test"):
        accuracy_am = AverageMeter('Accuracy', ':6.2f')
        precision_am = AverageMeter('Precision', ':6.2f')
        recall_am = AverageMeter('Recall', ':6.2f')
        f1_am = AverageMeter('F1', ':6.2f')
        all_preds = []
        all_targets = []
        self.eval()
        self.no_grad = True
        output_probs = []
        for i, (input, target) in enumerate(loader):
            input = input.to(self.device)
            target = target.to(self.device).view(-1).long()
            with torch.no_grad():
                output = self(input)
                loss = torch.sqrt(criterion(output, target))

            _, predicted = torch.max(output.data, 1)
            accuracy = accuracy_score(predicted.data.cpu(), target.data.cpu())
            accuracy_am.update(accuracy, input.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            tp_output_probs = torch.nn.functional.softmax(output.data, 1).detach().cpu().numpy()
            tp_ground_truth = target.data.cpu().numpy()
            tp_concatenated = numpy.concatenate((tp_output_probs, tp_ground_truth.reshape(-1, 1)), axis=1)
            output_probs.extend(tp_concatenated.tolist())

        precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        auc_ = roc_auc_score(y_true=all_targets, y_score=all_preds)
        cr = classification_report(all_targets, all_preds, target_names=["Benign", "SlowDoS"])

        rc_precision, rc_recall, rc_thresholds = precision_recall_curve(all_targets, numpy.array(output_probs)[:, 1])
        pr_auc = auc(rc_recall, rc_precision)

        #---------------------------------------
        # Step 7: Plot the Precision-Recall curve.
        plt.plot(rc_recall, rc_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig(f"{get_base_dir()}/pr_auc.jpg")
        #plt.show()
        # ---------------------------------------

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False

        string_csv = ""
        for line in output_probs:
            string_csv += ";".join([str(x) for x in line]) + "\n"

        with open(f"{get_base_dir()}/output_probs_{evaluation_on}.csv", "w") as f:
            f.write(string_csv)

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg, auc_, cr, pr_auc

    def fit(self, epochs, optimizer, criterion, train_loader, test_loader: Optional[DataLoader] = None):
        train_losses_per_epoch = []
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in tqdm(range(epochs)):
            avg_loss = self._train_epoch(train_loader, optimizer, criterion)
            train_losses_per_epoch.append(avg_loss)
            if test_loader is not None:
                accuracy, precision, recall, f1, auc_, cr, pr_auc = self.evaluate(test_loader, criterion)

        print("Finished training CPVAE!")
        if test_loader is not None:
            print("Final results:")
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc_}, pr_auc: {pr_auc}")
        #self.plotLoss(train_losses_per_epoch)

    def plotLoss(self, loss):
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        #plt.show()

# -----train and test-----#

# da: PlayItStraiht di Francesco Scala
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
