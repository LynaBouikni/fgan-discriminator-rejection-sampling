import torch
import os

def D_train(x, G, D, D_optimizer, divergence_fn):
    # =======================Train the discriminator======================= #
    D.zero_grad()

    # Train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output_real = D(x_real)
    D_real_loss = divergence_fn(D_output_real, y_real)
    D_real_score = D_output_real

    # Train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output_fake = D(x_fake)
    D_fake_loss = divergence_fn(D_output_fake, y_fake)
    D_fake_score = D_output_fake

    # Gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()


def G_train(x, G, D, G_optimizer, divergence_fn):
    # =======================Train the generator======================= #
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()

    G_output = G(z)
    D_output = D(G_output)
    G_loss = divergence_fn(D_output, y)

    # Gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()


def f_divergence(output, target):
    # Custom divergence function (replace with your desired function)
    return -torch.mean(target * torch.log(output) + (1 - target) * torch.log(1 - output))



def save_models(G, D, folder, div, eps):
    torch.save(G.state_dict(), os.path.join(folder, f'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, f'D.pth'))


def load_model(G, folder, filename):
    ckpt = torch.load(os.path.join(folder,filename))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
