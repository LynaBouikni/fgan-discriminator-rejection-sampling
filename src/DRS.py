import torch
import numpy as np


# Function to load a model from a checkpoint
def load_model(model_class, checkpoint_path, input_dim):
    model = model_class(input_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Loading the trained generator and discriminator
generator = load_model(Generator, 'path/to/G_BCE_50_LRMOD.pth', g_output_dim)  # Replace with your generator output dimension
discriminator = load_model(Discriminator, 'path/to/D_BCE_50_LRMOD.pth', d_input_dim)  # Replace with your discriminator input dimension

# Estimate M
M_estimated = estimate_M(generator, discriminator)

# Now we can use M_estimated in the rejection sampling process


def compute_acceptance_probabilities(discriminator, samples, M):
    # Compute the discriminator outputs for the batch of samples
    d_outputs = discriminator(samples).detach()

    # Calculate the acceptance probabilities for each sample in the batch
    acceptance_probs = torch.exp(d_outputs - M)
    return acceptance_probs

def rejection_sampling(generator, discriminator, M, num_samples, batch_size):
    accepted_samples = []
    while len(accepted_samples) < num_samples:
        # Generate a batch of samples
        noise = torch.randn(batch_size, 100)  # Assuming the input dimension for generator is 100
        generated_samples = generator(noise)

        # Compute acceptance probabilities for the batch
        acceptance_probs = compute_acceptance_probabilities(discriminator, generated_samples, M)

        # Decide whether to accept or reject each sample in the batch
        for i in range(batch_size):
            if torch.rand(1) < acceptance_probs[i]:
                accepted_samples.append(generated_samples[i].numpy())

            # Stop if we have enough samples
            if len(accepted_samples) >= num_samples:
                break

    return np.array(accepted_samples)



# M is estimated from samples.


def estimate_M(generator, discriminator, num_samples=10000):
    with torch.no_grad():  # No need to track gradients for this part
        # Generate a large batch of samples
        noise = torch.randn(num_samples, 100)  # A the input dimension for generator is 100
        generated_samples = generator(noise)

        # Pass the samples through the discriminator
        d_outputs = discriminator(generated_samples)

        # The discriminator outputs are used as a proxy for the ratio pd(x)/pg(x)
        # M is estimated as the maximum of these outputs
        M = torch.max(d_outputs).item()

    return M

#generator = Generator(g_output_dim=...)  # Initialize with the parameters
#discriminator = Discriminator(d_input_dim=...)  # Initialize with the parameters


# Specifying the batch size
batch_size = 32

# Generating samples using DRS
drs_samples = rejection_sampling(generator, discriminator, M, num_samples=100, batch_size=batch_size)
