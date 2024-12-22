import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae_models import *

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train IWAE or VAE on MNIST or Omniglot")
    parser.add_argument('--model', choices=['IWAE', 'VAE'], required=True, help="Model type: IWAE or VAE")
    parser.add_argument('-k', type=int, required=True, help="Number of importance samples")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('--dataset', choices=['MNIST', 'Omniglot'], required=True, help="Dataset: MNIST or Omniglot")
    return parser.parse_args()


def get_dataset(dataset_name, batch_size):
    # Binarize transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a 784-dimensional vector
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name == "Omniglot":
        dataset = datasets.Omniglot(root="./data", background=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    args = parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    dataloader = get_dataset(args.dataset, args.batch_size)

    # Initialize model
    model = IWAE(200, 100, 784).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Training loop
    save_path = "./results/"
    os.makedirs(save_path, exist_ok=True)
    loss_log = []

    for epoch in range(1, 101):  # Example: 100 epochs
        epoch_loss = 0.0
        model.train()
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()

            loss = model.compute_loss(data, args.k, args.model)
  
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            

        epoch_loss /= len(dataloader)
        loss_log.append(epoch_loss)

        # Log training loss
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
        if epoch % 100 == 99:
                torch.save(model.state_dict(), os.path.join(save_path, f"{args.model}_{args.dataset}_k_{args.k}_epoch_{epoch}.model"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, f"{args.model}_{args.dataset}_k_{args.k}.model"))

    # Save loss log
    with open(os.path.join(save_path, f"train_loss_{args.model}_{args.dataset}_k_{args.k}.txt"), "w") as f:
        for loss in loss_log:
            f.write(f"{loss}\n")

if __name__ == "__main__":
    main()
