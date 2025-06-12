import torch
import torchvision
import os
import argparse
from model import Generator, Discriminator
from utils import load_model
import matplotlib.pyplot as plt
import numpy as np


def compute_acceptance_probabilities(discriminator, samples, M, gamma):
    # Compute the discriminator logits for the batch of samples
    d_outputs = discriminator(samples).detach()
    # Compute F(x) using the method described in the text
    F_x = d_outputs - M - torch.log(1 - torch.exp(d_outputs - M) + 1e-8) - gamma
    # Calculate the acceptance probabilities for each sample in the batch
    acceptance_probs = torch.sigmoid(F_x)  # Using sigmoid as it's equivalent to 1 / (1 + exp(-F(x)))
    return acceptance_probs


def estimate_M(generator, discriminator, num_samples=10000):
    with torch.no_grad():
        noise = torch.randn(num_samples, 100).cuda()
        generated_samples = generator(noise)
        d_outputs = discriminator(generated_samples)
        M = torch.max(d_outputs).item()
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Discriminator Rejection Sampling implementation.')
    parser.add_argument("--batch_size", type=int, default=2048, help="The batch size to use for training.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Hyperparameter to modulate acceptance probability.")
    parser.add_argument("--generate_grid", action='store_true', help="Generate and display a grid of 20 examples.")
    args = parser.parse_args()

    # Model loading and generation of samples would remain the same as in the original script
    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    folder = 'checkpoints'

    # Load Generator
    generator = Generator(g_output_dim=mnist_dim).cuda()
    generator = load_model(generator, folder, 'G.pth')
    generator = torch.nn.DataParallel(generator).cuda()
    generator.eval()

    # Load Discriminator
    discriminator = Discriminator(d_input_dim=mnist_dim).cuda()
    discriminator = load_model(discriminator, folder, 'D.pth')
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    discriminator.eval()

    print('Models loaded.')

    # Estimate M
    M_estimated = estimate_M(generator, discriminator)
    gamma = args.gamma  # Modulate acceptance probability

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    if args.generate_grid:
        grid_samples = []

    max_discriminator_output = -float('inf')  # Initialize with negative infinity
    with torch.no_grad():
        while n_samples < 10000:
            z = torch.randn(args.batch_size, 100).cuda()
            generated_samples = generator(z)
            d_outputs = discriminator(generated_samples).detach()
            max_discriminator_output = max(max_discriminator_output, torch.max(d_outputs).item())
            M_dynamic = max(max_discriminator_output, M_estimated)  # Use the larger of the two M values

            acceptance_probs = compute_acceptance_probabilities(discriminator, generated_samples, M_dynamic, gamma)

            generated_samples = generated_samples.reshape(args.batch_size, 28, 28)
            for k in range(generated_samples.shape[0]):
                if 0.8 < acceptance_probs[k] and n_samples < 10000:
                    torchvision.utils.save_image(generated_samples[k:k + 1],
                                                 os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
                    if args.generate_grid and len(grid_samples) < 20:
                        grid_samples.append(generated_samples[k].cpu().numpy())

        if args.generate_grid:
            grid_images = np.concatenate(grid_samples, axis=1)
            grid_image_path = os.path.join('samples', 'GRID_DRS.png')
            plt.imsave(grid_image_path, grid_images, cmap='gray')
            plt.show()
            print("Grid saved")

    print('Generation completed.')
