import nn_layers.vp_layers as vp
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
from classifier_models.utils import train_classifier, run_classifier, return_model_accuracy
from dataloaders import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio

class UnknownLayer(Exception):
    pass

def hook_fn(m, i, o):
    print(i)

def add_hook(net):
    hook_handles = []
    for m in net.modules():
        handle = m.register_forward_hook(hook_fn)
        hook_handles.append(handle)
    return hook_handles

def load_weights(name, path):
    # Modified to parse plain text accuracy log file
    model_path = os.path.join(path, f'{name}.dat')
    results = []
    with open(model_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "Train accuracy" in line and "Test accuracy" in line:
            parts = line.strip().split(", ")
            epoch = int(parts[0].split(":")[1].replace("]", ""))
            train_acc = float(parts[1].split(":")[1].replace("%", ""))
            test_acc = float(parts[2].split(":")[1].replace("%", ""))
            results.append({"epoch": epoch, "train_accuracy": train_acc, "test_accuracy": test_acc})
    return results

# The rest of your code expects a model, but output.dat is just logs.
# So, let's just print the parsed results for demonstration.

path = 'C:\\Users\\DELL\\VPSNN\\notebooks'
name = 'output'
results = load_weights(name, path)

# Example: print all epochs' accuracies
for entry in results:
    print(f"Epoch {entry['epoch']}: Train accuracy = {entry['train_accuracy']}%, Test accuracy = {entry['test_accuracy']}%")

# If you want to plot the accuracies:
epochs = [entry['epoch'] for entry in results]
train_acc = [entry['train_accuracy'] for entry in results]
test_acc = [entry['test_accuracy'] for entry in results]

plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Test Accuracy per Epoch')
plt.show()



