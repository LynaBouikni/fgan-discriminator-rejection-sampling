import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from utils import D_train, G_train, save_models
import torch
from torchvision.models import inception_v3
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os


def get_generated_dataloader(generator, batch_size, noise_dim=100, num_samples=10000):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, device='cuda')
        generated_images = generator(noise)
    generator.train()
    generated_images = generated_images.detach().cpu()
    generated_dataset = TensorDataset(generated_images)
    return DataLoader(generated_dataset, batch_size=batch_size)


def plot_losses(d_losses, g_losses, divergence_type):
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss During Training ({divergence_type})")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'loss_plot_{divergence_type}.png')
    plt.show()


def extract_features(model, dataloader):
    features = []
    for images, _ in dataloader:
        with torch.no_grad():
            pred = model(images.cuda())
        features.append(pred.cpu().detach().numpy())
    return np.concatenate(features, axis=0)


def compute_precision_recall(generator, real_dataloader, gen_dataloader, device='cuda'):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    real_features = extract_features(inception_model, real_dataloader)
    gen_features = extract_features(inception_model, gen_dataloader)
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    gmm.fit(real_features)
    gen_scores = gmm.score_samples(gen_features)
    precision = np.mean(gen_scores > np.log(0.5))
    sampled_features = gmm.sample(n_samples=len(gen_features))[0]
    recall = np.mean(gmm.score_samples(sampled_features) > np.log(0.5))
    return precision, recall


# define loss
EPSILON = 1e-8  # Small epsilon value

def f_divergence_function(divergence_type):
    if divergence_type == "BCE":
        return nn.BCELoss()
    elif divergence_type == "KLD":
        return lambda output, target: torch.mean(
            target * torch.log((target + EPSILON) / (output + EPSILON)) + (1 - target) * torch.log(
                ((1 - target) + EPSILON) / ((1 - output) + EPSILON)))
    elif divergence_type == "JS":
        return lambda output, target: 0.5 * (torch.mean(
            target * torch.log((target + EPSILON) / ((0.5 * (output + target)) + EPSILON))) + torch.mean(
            output * torch.log(((0.5 * (output + target)) + EPSILON) / (output + EPSILON))))
    else:
        raise ValueError(f"Unsupported divergence type: {divergence_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN Model.')
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="Interval for evaluating models.")
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    divergences = ['KLD']

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize models
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # Training loop
    for i in range(0, len(divergences)):

        div_code = divergences[i]
        criterion = f_divergence_function(div_code)
        error_count = 0
        error_flag = False

        # define optimizers
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

        print('Start Training :', div_code)

        n_epoch = args.epochs
        for epoch in range(1, args.epochs + 1):
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader),
                                          desc=f"Epoch {epoch}/{args.epochs}"):
                x = x.view(-1, mnist_dim).to(device)
                z = torch.randn(args.batch_size, 100, device=device)

                # Train discriminator and generator
                D_loss = D_train(x, G, D, D_optimizer, criterion)
                G_loss = G_train(z, G, D, G_optimizer, criterion)

                D_losses.append(D_loss)
                G_losses.append(G_loss)

                if math.isnan(D_loss) or math.isnan(G_loss):
                    error_flag = True

            # If loss is NAN, we restart with that divergence
            if error_flag:
                error_count += 1
                if error_count < 8:
                    i -= 1  # We only allow 8 errors, and we move on
                else:
                    print(f'DIVERGENCE {div_code} ABORTED: REACHED 8 ERRORS')
                break

            # Save models every 25 epochs
            if epoch % 25 == 0:
                save_models(G, D, 'checkpoints', div_code, epoch)

            # Plot losses
            #if epoch % args.evaluation_interval == 0:
                #plot_losses(D_losses, G_losses, div_code)

                # Evaluate using precision and recall
                #gen_dataloader = get_generated_dataloader(G, args.batch_size)
                #precision, recall = compute_precision_recall(G, DataLoader(train_dataset, batch_size=args.batch_size,
                                                                           # shuffle=False), gen_dataloader, device)
                #print(f'Epoch {epoch} - Precision: {precision:.4f}, Recall: {recall:.4f}')

        print('Training done:', div_code)

    print('ALL TRAINING DONE')
