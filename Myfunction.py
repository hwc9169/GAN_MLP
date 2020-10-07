import matplotlib.pyplot as plt
import torchvision.utils as utils
import numpy as np
import torch 

def imshow(img):
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()


def image_grid(img):
    img = utils.make_grid(img)
    img = img.cpu().detach().numpy()
    plt.imshow(np.transpose(img, (1,2,0)), cmap='gray')
    plt.show()


def sample_z(batch_size=1, d_noise=100, device='cpu'):
    return torch.randn(batch_size, d_noise, device=device)


def train(G, G_optimizer, D, D_optimizer, train_data_loader, device='cpu'):
    G.train()
    D.train()

    for image_batch, label_batch in train_data_loader:
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        
        D_optimizer.zero_grad()
   
        p_real = D(image_batch.view(-1, 28 * 28))
        p_fake = D(G(sample_z(image_batch.shape[0], device=device)))

        D_loss = -1 * (torch.log(p_real) + torch.log(1-p_fake)).mean()

        D_loss.backward()
        D_optimizer.step()

        G_optimizer.zero_grad()

        p_fake = D(G(sample_z(image_batch.shape[0], device=device)))
    
        G_loss = -1 * torch.log(p_fake).mean()

        G_loss.backward()
        G_optimizer.step()


def evaluate(G, D, test_data_loader, device):
    p_real = 0
    p_fake = 0

    D.eval()
    G.eval()

    for image_batch, label_batch in test_data_loader:
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        with torch.autograd.no_grad():
            p_real += torch.sum(D(image_batch.view(-1, 28*28)))/10000
            p_fake += torch.sum(D(G(sample_z(image_batch.shape[0], device=device))))/10000

    return p_real, p_fake

def init_weight(model):
    for weight in model.parameters(): 
        if weight.dim() > 1:
            torch.nn.init.kaiming_normal_(weight)
        
        else:
            torch.nn.init.uniform(weight, 0, 0.1)