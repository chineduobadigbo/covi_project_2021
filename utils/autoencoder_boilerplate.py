from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn

class PatchDataset(Dataset): #a dataset object has to be definied that specifies how PyTorch can access the training data

    def __init__(self, X): #Initialization
        self.X = X
    
    def __len__(self): #Denotes the total number of samples
        return len(self.X)

    def __getitem__(self, index): #Generates one sample of data
        image = self.X[index] # Select sample
        X = self.transform(image)
        return X

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        ])


class Autoencoder(nn.Module): #defines the autoencoder architecture
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #the stride parameter is describing how much the resoltion decreases in each step
            nn.Conv2d(3, 64, 3, stride=2, padding=1), # N, 64, 32, 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3 , stride=2, padding=1), # N, 128, 16, 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3 , stride=2, padding=1), # N, 256, 8, 8
            nn.ReLU(),
            #in the last step, we make the kernel size equal to the image resolution, thus reducing it to just one pixel
            nn.Conv2d(256, 512, 8) # N, 512, 1, 1 -> this results a simple 512d vector
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 8), # N, 256, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # N, 128, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # N, 64, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1), # N, 3, 64, 64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded