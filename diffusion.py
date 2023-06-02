import os
import torch
import math
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
# from unet import UNet
import matplotlib.pyplot as plt
import time

root_path = os.path.dirname(os.path.abspath(__file__))
target_image = root_path + '/test_images/01174-1941935081.png'
target_model_name = root_path + '/unet.pth'
device = 'cuda:0'
#device = 'cpu'

def get_image(name):
    return Image.open(name)

print(torch.__version__)
print(torch.cuda.is_available())

class DiffusionModel:
    def __init__(self, start_schedule=0.0001, end_schedule=0.02,timesteps = 300):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        
        self.betas = torch.linspace(start_schedule, end_schedule,timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def forward(self, x_0, t, device):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return mean + variance,noise.to(device)
    
    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise 
            return mean + variance
        
    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        result = values.gather(-1,t.cpu())
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# def forward_diffusion(x0, t, betas = torch.linspace(0.0,1.0,5)):
#     noise = torch.rand_like(x0) # random tensor with values sampled from N(0,1)
#     alphas = 1 - betas
#     alpha_hat = torch.cumprod(alphas,axis = 0)
#     alpha_hat_t = alpha_hat.gather(-1, t).reshape(-1,1,1,1)

#     mean = alpha_hat_t.sqrt() * x0
#     variance = torch.sqrt(1 - alpha_hat_t) * noise
#     return mean + variance,noise

# Define the transformation: Convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor(), # in range of 0~1
    transforms.Lambda(lambda t: (t*2) - 1) # to range -1~1
])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1,2,0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

pil_image = get_image(target_image)
torch_image = transform(pil_image)

diffusion_model = DiffusionModel()

NUM_DISPLAY_IMAGES = 5
torch_image_batch = torch.stack([torch_image] * NUM_DISPLAY_IMAGES)
t = torch.linspace(0, diffusion_model.timesteps - 1, NUM_DISPLAY_IMAGES).long()
noisy_image_batch, _ = diffusion_model.forward(torch_image_batch, t, device)

def show_batched_image_tensors(images):
    target_size = 64
    final_image = Image.new('RGB', (target_size * len(images), target_size))
    x = 0
    for img_tensor in images:
        img = reverse_transform(img_tensor)
        final_image.paste(img, (x, 0))
        x += target_size
    final_image.show()
    
# show_batched_image_tensors(noisy_image_batch)

def plot_noise_prediction(noise, predicted_noise):
    plt.figure(figsize=(15,15))
    f, ax = plt.subplots(1, 2, figsize = (5,5))
    ax[0].imshow(reverse_transform(noise))
    ax[0].set_title(f"ground truth noise", fontsize = 10)
    ax[1].imshow(reverse_transform(predicted_noise))
    ax[1].set_title(f"predicted noise", fontsize = 10)
    plt.show()

def plot_noise_distribution(noise, predicted_noise):
    plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.legend()
    plt.show()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, num_filters = 3, downsample=True):
        super().__init__()
        
        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)
        
        self.downsample = downsample
        
        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
            
        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)
        
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time[(..., ) + (None, ) * 2]
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label[(..., ) + (None, ) * 2]
            
        o = self.bnorm2(self.relu(self.conv2(o)))

        return self.final(o)
    
class UNet(nn.Module):
    def __init__(self, img_channels = 3, time_embedding_dims = 128, labels = False, sequence_channels = (64, 128, 256, 512, 1024)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        sequence_channels_rev = reversed(sequence_channels)
        
        self.downsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels,downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)

    
    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)
            
        return self.conv2(o)



# Train
def Train():
    unet = UNet().to(device)
    NUM_EPOCHS = 500
    PRINT_FREQUENCY = 400
    LearnRate = 0.001
    BATCH_SIZE = 128
    VERBOSE = False

    start_time = time.time()
    mean_losses = []
    optimizer = torch.optim.Adam(unet.parameters(), lr = LearnRate)
    for epoch in range(NUM_EPOCHS):
        mse_loss_epoch = []

        batch_size = torch.stack([torch_image] * BATCH_SIZE)
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
        noisy_image, gt_noise = diffusion_model.forward(batch_size, t, device)
        predicted_noise = unet(noisy_image, t)

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(gt_noise, predicted_noise)
        mse_loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()

        mean_loss = np.mean(mse_loss_epoch)
        mean_losses.append(mean_loss)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train Loss {mean_loss}")

        if VERBOSE and epoch % PRINT_FREQUENCY == 0:
            print("------")
            with torch.no_grad():
                plot_noise_prediction(gt_noise[0], predicted_noise[0])
                plot_noise_distribution(gt_noise, predicted_noise)
    torch.save(unet.state_dict(), target_model_name)

    indices = list(range(len(mean_losses)))
    plt.plot(indices, mean_losses)
    plt.show()
    run_time = time.time() - start_time
    print(f"Training take {run_time} seconds.")


# If NOT trained, train it
if not os.path.exists(target_model_name):
    Train()

# Sample
unet = UNet().to(device)
unet.load_state_dict(torch.load(target_model_name))

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)#固定随机种子(CPU)
    if torch.cuda.is_available(): # 固定随机种子(GPU)
        torch.cuda.manual_seed(seed)#为当前GPU设置
        torch.cuda.manual_seed_all(seed)#为所有GPU设置
    np.random.seed(seed)#保证后续使用random函数时,产生固定的随机数

same_seeds(1235)

diffusion_model = DiffusionModel()
tensor_images = []
img = torch.randn((1, 3, 64, 64)).to(device)
for i in reversed(range(diffusion_model.timesteps)):
    t = torch.full((1,), i, dtype = torch.long, device = device)
    print(t)
    img = diffusion_model.backward(img, t, unet.eval())
    if i % 60 == 0:
        print(i)
        print(img.size())
        img_show = reverse_transform(img[0])
        img_show.show()