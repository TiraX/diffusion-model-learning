# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Load the dataset
dir_path = os.path.dirname(os.path.abspath(__file__))
train_set = datasets.ImageFolder(dir_path + '/train_images', transform=transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))
test_set = datasets.ImageFolder(dir_path + '/test_images', transform=transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1,2,0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

# Split the dataset into training, validation, and testing sets
# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Set up DataLoaders for the sets
BATCH_SIZE = 4
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Define the network architecture for converting image to latent space
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     # Input: (3, 64, 64)
        #     nn.Conv2d(3, 32, 4, 2, 1),
        #     nn.LeakyReLU(0.2),
        #     # (32, 32, 32)
        #     nn.Conv2d(32, 64, 4, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #     # (64, 16, 16)
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2),
        #     # (128, 8, 8)
        #     nn.Conv2d(128, 256, 4, 2, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2),
        #     # (256, 4, 4)
        #     nn.Flatten(),
        #     nn.Linear(256 * 4 * 4, latent_dim),
        # )
        self.encoder = nn.Sequential(
            # Input: (3, 256, 256)
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # (32, 128, 128)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # (64, 64, 64)
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # (128, 32, 32)
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # (256, 16, 16)
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # (512, 8, 8)
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            # (1024, 4, 4)
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

 # Define the network architecture for converting latent space back to image space
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: (latent_dim)
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.ReLU(),
            # (1024, 4, 4)
            nn.Unflatten(1, (1024, 4, 4)),
            # (1024, 4, 4)
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # (512, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (256, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (128, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (64, 64, 64)
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (32, 128, 128)
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
            # Output: (3, 256, 256)
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = AutoEncoder(latent_dim=256)

# Train the model
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')
model.to(device)

criterion = nn.MSELoss()

target_model_name = dir_path + '/auto_encoder.pth'
if not os.path.exists(target_model_name):
    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            img = data[0].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass to get outputs
            outputs = model(img)  # This now gives reconstructed images
            # compute the loss
            loss = criterion(outputs, img)
            # backward pass to compute gradient
            loss.backward()
            # update weights
            optimizer.step()
            total_loss += loss.item()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / len(train_loader)))

    # Save the model
    torch.save(model.state_dict(), target_model_name)
else:
    model.load_state_dict(torch.load(target_model_name))

def show_batched_image_tensors(images):
    target_size = 256
    final_image = Image.new('RGB', (target_size * len(images), target_size))
    x = 0
    for img_tensor in images:
        img = reverse_transform(img_tensor)
        final_image.paste(img, (x, 0))
        x += target_size
    final_image.show()

# Test the model
model.eval()
total_loss = 0
index = 0
with torch.no_grad():
    for data in test_loader:
        img = data[0].to(device)
        outputs = model(img)
        if index == 0:
            show_batched_image_tensors(img)
            show_batched_image_tensors(outputs)
        loss = criterion(outputs, img)
        total_loss += loss.item()
        index += 1

print('Test Loss: {:.4f}'.format(total_loss / len(test_loader)))