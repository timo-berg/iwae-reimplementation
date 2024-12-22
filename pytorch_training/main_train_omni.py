import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")
from vae_models import *
from sys import exit
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_omniglot_dataloader(batch_size=1000, train=True):
    """
    Load the Omniglot dataset and return a DataLoader.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): If True, loads the training split; otherwise, loads the test split.

    Returns:
        DataLoader: PyTorch DataLoader for Omniglot.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a 784-dimensional vector
    ])

    omniglot_dataset = datasets.Omniglot(
        root='./data',
        background=train,  # background=True -> training set; background=False -> testing set
        transform=transform,
        download=True
    )

    dataloader = DataLoader(omniglot_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

parser = argparse.ArgumentParser(description="Importance Weighted Auto-Encoder")
parser.add_argument("--model", type = str,
                    choices = ["IWAE", "VAE"],
                    required = True,
                    help = "choose VAE or IWAE to use")
parser.add_argument("--num_stochastic_layers", type = int,
                    choices = [1, 2],
                    required = True,
                    help = "num of stochastic layers used in the model")
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of
                              ELBO when using VAE; num of samples used in
                              importance weighted ELBO when using IWAE.""")
args = parser.parse_args()

batch_size = 256
train_data_loader = get_omniglot_dataloader(batch_size=batch_size, train=True)
test_data_loader = get_omniglot_dataloader(batch_size=batch_size, train=False)

# Inspect one batch
for images, labels in train_data_loader:
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    break


if args.num_stochastic_layers == 1:
    vae = IWAE_1(50, 784)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(100, 50, 784)

vae.double()
vae.cuda()

optimizer = optim.Adam(vae.parameters())
num_epoches = 500
train_loss_epoch = []
for epoch in range(num_epoches):
    running_loss = []
    for idx, feat_and_label in enumerate(train_data_loader):
        data, labels = feat_and_label
        data = data.double()
        inputs = Variable(data).cuda()
        # Get the actual size of the current batch
        current_batch_size = inputs.size(0)
        if args.model == "IWAE":
            inputs = inputs.expand(args.num_samples, current_batch_size, 784)
        elif args.model == "VAE":
            inputs = inputs.repeat(args.num_samples, 1)
            inputs = inputs.expand(1, current_batch_size*args.num_samples, 784)

        optimizer.zero_grad()
        loss = vae.train_loss(inputs)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    print(("Epoch: {:>4}, loss: {:>4.2f}")
            .format(epoch, loss.item()), flush = True)


    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 100 == 0:
        torch.save(vae.state_dict(),
                   ("./output_omni/model/{}_layers_{}_k_{}_epoch_{}_small_batch.model")
                   .format(args.model, args.num_stochastic_layers,
                           args.num_samples, epoch))

torch.save(vae.state_dict(),
            ("./output_omni/model/{}_layers_{}_k_{}_small_batch.model")
            .format(args.model, args.num_stochastic_layers,
                        args.num_samples))

## save training loss to txt file
with open("./output_omni/train_loss_{}_layers_{}_k_{}_small_batch.txt"
          .format(args.model, args.num_stochastic_layers,
                  args.num_samples), 'w') as file_handle:
    for loss in train_loss_epoch:
        file_handle.write("{}\n".format(loss))
