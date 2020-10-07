from Myfunction import *
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils as utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_noise = 100
batch_size = 200
epochs = 200
standardizator = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5),
                        std=(0.5))])

train_data = datasets.MNIST(root='data/', download=True, train=True, transform=standardizator)
test_data = datasets.MNIST(root='data/', download=True, train=False, transform=standardizator)

train_data_loader = utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_data_loader = utils.data.DataLoader(test_data, batch_size, shuffle=True)

img, label = next(iter(train_data_loader))
image_grid(img[0:12])

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.fc1 = nn.Linear(d_noise, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 28*28)
        self.tanh = nn.Tanh()


    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x


D = nn.Sequential(
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(device)


z = sample_z(device=device)
G = G().to(device)
init_weight(G)
init_weight(D)

criterion = nn.BCELoss()

G_optimizer = optim.Adam(G.parameters(), lr=0.0001)
D_optimizer = optim.Adam(D.parameters(), lr=0.0001)

for epoch in range (epochs):
    train(G, G_optimizer, D, D_optimizer, train_data_loader, device)
    
    if (epoch+1)%50 == 0:
        p_real, p_fake = evaluate(G, D, test_data_loader, device)
        print("Epoch : {:4d}/{:d} p_real : {:.6f} p_fake : {:.6f}".format(epoch+1, epochs, p_real, p_fake))
        image_grid(G(sample_z(16, device=device)).view(-1,1,28,28))

