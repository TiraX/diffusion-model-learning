# Import PyTorch
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

LATENT_DIM = 512

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # Additional Conv layer for 128x128 images
        #self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        #self.bn5 = nn.BatchNorm2d(512) # BatchNorm for additional layer
        self.fc_mu = nn.Linear(256*8*8, LATENT_DIM)
        self.fc_log_var = nn.Linear(256*8*8, LATENT_DIM)
        self.dropout = nn.Dropout(0.25) # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        #x = F.relu(self.bn5(self.conv5(x))) # Additional Conv layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x) # Dropout after flatten
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(LATENT_DIM, 256*8*8)
        #self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # Additional Transposed Conv layer
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        #self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25) # Dropout for regularization

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x) # Dropout after fc
        x = x.view(x.size(0), 256, 8, 8) # Additional Transposed Conv layer
        #x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x)) # Ensuring output is in the range [0,1]
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, log_var):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     return BCE + KLD
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# transform = transforms.Compose([
#     transforms.ToTensor(), # in range of 0~1
#     transforms.Lambda(lambda t: (t*2) - 1) # to range -1~1
# ])

# Load the dataset
TARGET_SIZE = 128
dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = 'C:\\train_dataset'
train_set = datasets.ImageFolder(dir_path + '/train_images', transform=transforms.Compose([
    #transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))

test_set = datasets.ImageFolder(dir_path + '/test_images', transform=transforms.Compose([
    #transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]))

reverse_transform = transforms.Compose([
    #transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1,2,0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

# Set up DataLoaders for the sets
BATCH_SIZE = 4
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

best_model_name = dir_path + '/vae_best.pth'
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    model.eval()  # set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # do not compute gradients
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

if not os.path.exists(best_model_name):
    NUM_EPOCHS = 100
    min_val_loss = np.inf
    patience = 10
    for epoch in range(1, NUM_EPOCHS + 1):
        train(epoch)
        val_loss = test()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), best_model_name)
            print(best_model_name + ' saved...')
            wait = 0
        else:
            wait += 1
            if wait > patience:
                print("Early stopping")
                break


model = VAE().to(device)
model.load_state_dict(torch.load(best_model_name))
print(f'{best_model_name} loaded...')

def show_batched_image_tensors(images):
    target_size = TARGET_SIZE
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
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        total_loss += loss.item()
        #if index == 0:
        show_batched_image_tensors(data)
        show_batched_image_tensors(recon_batch)
        index += 1

print('Test Loss: {:.4f}'.format(total_loss / len(test_loader.dataset)))