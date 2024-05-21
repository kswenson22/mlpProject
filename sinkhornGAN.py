import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch_fidelity import calculate_metrics
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from PIL import Image
import pandas as pd
import os
import torch.nn.functional as F
import multiprocessing
import joblib
from joblib import load
import torchvision.models as models
from geomloss import SamplesLoss
from torchvision.utils import save_image
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# seed for reproducibility
seed_value = 2024
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.mps.manual_seed(seed_value)
device = torch.device('mps')

# Setup checkpoint directory and metrics log file
checkpoint_dir = '/Users/kieran/Documents/mlpProject/monet2photo'
metrics_log_file = os.path.join(checkpoint_dir, 'training_metrics.csv')
os.makedirs(checkpoint_dir, exist_ok=True)

# Define checkpointing frequency
checkpoint_freq = 10

# Function to save checkpoints
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

# Function to update metrics log
def update_metrics_log(metrics, filename):
    df = pd.DataFrame([metrics])
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='a', index=False)
    else:
        df.to_csv(filename, mode='a', index=False, header=False)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Encoder layers
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # Residual blocks
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # Decoder layers
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

def main():
    print("Main function")
 # Now you can initialize the networks
    input_nc = 3  # Number of input channels
    output_nc = 3  # Number of output channels
    # Initialize networks
    G_AB = Generator(input_nc, output_nc).to(device)
    G_BA = Generator(output_nc, input_nc).to(device)
    D_A = Discriminator(input_nc).to(device)
    D_B = Discriminator(output_nc).to(device)
    print("Initialized networks")

    # Define optimizer
    print("Optimizer definition")
    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Define data loaders
    print("Data loaders")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset_A = ImageFolder(root="/Users/kieran/Documents/mlpProject/monet/photo_jpg", transform=transform)
    dataset_B = ImageFolder(root="/Users/kieran/Documents/mlpProject/monet/monet_jpg", transform=transform)
    # test_A = ImageFolder(root="/Users/kieran/Documents/mlpProject/zebraHorse/testHorses", transform=transform)
    # test_B = ImageFolder(root="/Users/kieran/Documents/mlpProject/zebraHorse/testZebras", transform=transform)
    dataloader_A = DataLoader(dataset_A, batch_size=8, shuffle=True, num_workers=6, drop_last=True)
    dataloader_B = DataLoader(dataset_B, batch_size=8, shuffle=True, num_workers=6, drop_last=True)

    # # dataloaders for image saving
    # dataloader_A_test = DataLoader(dataset_A, batch_size=5, shuffle=False, num_workers=5)
    # dataloader_B_test = DataLoader(dataset_B, batch_size=5, shuffle=False, num_workers=5)

    num_epochs = 100
    lambda_identity = 5.0
    lambda_cycle = 10.0
    lambda_sinkhorn = 0

    adversarial_loss = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")  

        # Reset total losses for each epoch
        total_cycle_loss = 0.0
        total_sinkhorn_loss = 0.0
        total_identity_loss = 0.0
        total_loss_G_AB = 0.0
        total_loss_G_BA = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0
        total_sinkhorn_loss = 0.0
        total_loss_G = 0.0

        for i, data in enumerate(zip(dataloader_A, dataloader_B)):
            real_A, _ = data[0]
            real_B, _ = data[1]
            if i % 100 == 0:
                print("Data loop: ", i)
            # Move real_A and real_B to the same device as the model
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            # Train the generators
            optimizer_G.zero_grad()

            fake_B = G_AB(real_A)  # Generate fake image B from real image A
            rec_A = G_BA(fake_B)  # Reconstruct image A from fake image B
            fake_A = G_BA(real_B)  # Generate fake image A from real image B
            rec_B = G_AB(fake_A)  # Reconstruct image B from fake image A

            rec_A = rec_A.to(device)
            rec_B = rec_B.to(device)

            # Identity loss
            identity_loss = (torch.mean(torch.abs(real_A - fake_A)) + torch.mean(torch.abs(real_B - fake_B))) * lambda_identity

            # Adversarial loss
            pred_fake_B = torch.sigmoid(D_B(fake_B)) 
            loss_G_AB = adversarial_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            pred_fake_A = torch.sigmoid(D_A(fake_A))
            loss_G_BA = adversarial_loss(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle consistency loss
            cycle_loss = (torch.mean(torch.abs(real_A - rec_A)) + torch.mean(torch.abs(real_B - rec_B))) * lambda_cycle

            # Sinkhorn loss
            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9)

            fake_A_reshaped = fake_A.view(fake_A.shape[0], -1).to(device)
            fake_B_reshaped = fake_B.view(fake_B.shape[0], -1).to(device)
            real_A_reshaped = real_A.view(real_A.shape[0], -1).to(device)
            real_B_reshaped = real_B.view(real_B.shape[0], -1).to(device)

            sinkhorn_divergence = (sinkhorn_loss(fake_A_reshaped, real_A_reshaped) + sinkhorn_loss(fake_B_reshaped, real_B_reshaped))*lambda_sinkhorn

            # Total generator loss
            loss_G = loss_G_AB + loss_G_BA + cycle_loss + identity_loss + sinkhorn_divergence
            loss_G.backward()
            optimizer_G.step()

            # Accumulate losses
            total_loss_G_AB += loss_G_AB.item()
            total_loss_G_BA += loss_G_BA.item()
            total_cycle_loss += cycle_loss.item()
            total_identity_loss += identity_loss.item()
            total_sinkhorn_loss += sinkhorn_divergence.item()
            total_loss_G += loss_G.item()

            # Train the discriminators
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            pred_real_A = torch.sigmoid(D_A(real_A))
            loss_D_real_A = adversarial_loss(pred_real_A, torch.ones_like(pred_real_A))

            pred_fake_A = torch.sigmoid(D_A(fake_A.detach()))
            loss_D_fake_A = adversarial_loss(pred_fake_A, torch.zeros_like(pred_fake_A))

            pred_real_B = torch.sigmoid(D_B(real_B))
            loss_D_real_B = adversarial_loss(pred_real_B, torch.ones_like(pred_real_B))

            pred_fake_B = torch.sigmoid(D_B(fake_B.detach()))
            loss_D_fake_B = adversarial_loss(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            loss_D_A.backward()
            loss_D_B.backward()

            optimizer_D_A.step()
            optimizer_D_B.step()

             # Accumulate losses
            total_loss_D_A += loss_D_A.item()
            total_loss_D_B += loss_D_B.item()

        # Calculate average losses for the epoch
        avg_loss_G_AB = total_loss_G_AB / len(dataloader_A)
        avg_loss_G_BA = total_loss_G_BA / len(dataloader_B)
        avg_loss_D_A = total_loss_D_A / len(dataloader_A)
        avg_loss_D_B = total_loss_D_B / len(dataloader_A)
        avg_cycle_loss = total_cycle_loss / len(dataloader_A)
        avg_sinkhorn_loss = total_sinkhorn_loss / len(dataloader_A)
        avg_identity_loss = total_identity_loss / len(dataloader_A)

        # Print or log the average losses for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Generator AB Loss: {avg_loss_G_AB:.4f}, "
            f"Generator BA Loss: {avg_loss_G_BA:.4f}, "
            f"Discriminator A Loss: {avg_loss_D_A:.4f}, "
            f"Discriminator B Loss: {avg_loss_D_B:.4f},"
            f"Cycle Loss: {avg_cycle_loss:.4f}, "
            f"Identity Loss: {avg_identity_loss:.4f}"
            f"Sinkhorn Loss: {avg_sinkhorn_loss:.4f}, ")
        
        # metrics dictionary
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss_G_AB': total_loss_G_AB / len(dataloader_A),
            'loss_G_BA': total_loss_G_BA / len(dataloader_A),
            'loss_D_A': total_loss_D_A / len(dataloader_A),
            'loss_D_B': total_loss_D_B / len(dataloader_A),
            'Sinkhorn Loss': total_sinkhorn_loss / len(dataloader_A),
            'Cycle Loss': total_cycle_loss / len(dataloader_A),
            'Identity Loss': total_identity_loss / len(dataloader_A)
        }
        update_metrics_log(epoch_metrics, metrics_log_file)

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch in [0,1,2,3,4,5] or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch,
                'G_AB_state_dict': G_AB.state_dict(),
                'G_BA_state_dict': G_BA.state_dict(),
                'D_A_state_dict': D_A.state_dict(),
                'D_B_state_dict': D_B.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
                'learning_rate_G': optimizer_G.param_groups[0]['lr'],
                'learning_rate_D_A': optimizer_D_A.param_groups[0]['lr'],
                'learning_rate_D_B': optimizer_D_B.param_groups[0]['lr'],
                'rng_state_pytorch': torch.get_rng_state(),
                'metrics': epoch_metrics  # Save metrics along with model
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Ensure multiprocessing works in frozen executables
    device = torch.device('mps')
    main().to(device)
    print(device)

