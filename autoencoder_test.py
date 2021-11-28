import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

num_epochs = 10

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn. Conv2d(1, 16, 3, stride=2, padding=1), # N, 16, 14, 14
            nn. ReLU() ,
            nn. Conv2d(16, 32, 3 , stride=2, padding=1), # N, 32, 7, 7
            nn. ReLU(),
            nn. Conv2d(32, 64, 7) # N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn. ConvTranspose2d(64, 32, 7), # N, 32, 7, 7
            nn. ReLU() ,
            nn. ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16,14,14
            nn. ReLU() ,
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
            nn. Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def loadData():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./mnistdata',train=True,download=True,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size=64,shuffle=True)
    return data_loader

def trainEncoder(data_loader):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model. parameters (), lr=1e-3, weight_decay=1e-5)

    outputs = []
    for epoch in range (num_epochs) :
        for (img, _) in data_loader:
            #img = img. reshape (-1, 28*28)
            recon = model (img)
            loss = criterion (recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print (f' Epoch: {epoch+1}, Loss: {loss. item ():.4f}')
        outputs.append((epoch, img, recon))

    return outputs



def displayResults(outputs):
    for k in range (0, num_epochs, 4):
        plt. figure (figsize=(9, 2))
        plt. gray ()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate (imgs):
            if i >= 9: break
            plt. subplot (2, 9, i+1)
            plt.imshow(item[0])
        for i, item in enumerate(recon) :
            if i >= 9: break
            plt. subplot(2, 9, 9+i+1) # row length + i + 1
            plt.imshow(item[0])

    plt.show()


if __name__ == "__main__":
    data_loader=loadData()
    outputs = trainEncoder(data_loader)
    displayResults(outputs)