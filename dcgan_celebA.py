import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid

from torch.utils.data import DataLoader

import numpy as np
import math
import os, time
import matplotlib.pyplot as plt
import itertools

import pickle
import imageio

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_cuda(x):
    return x.to(DEVICE)

fixed_z = to_cuda(torch.randn(5*5, 100))

def save_sample(epoch, path = 'result.png'):
    G.eval()
    test_img = G(fixed_z)
    G.train()
    
    grid = 5
    fig, ax = plt.subplots(grid, grid, figsize = (5, 5))
    for i, j in itertools.product(range(grid), range(grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
        
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_img[k].cpu().data.numpy().transpose(1, 2, 0) + 1 ) / 2)
        
    label = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, label, ha = 'center')
    
    plt.savefig(path)
    plt.close()

def save_train_hist(hist, path = 'train_hist.png'):
    x = range(len(hist['D_losses']))
    
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    
    plt.plot(x, y1, label = 'D_loss')
    plt.plot(x, y2, label = 'G_loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend(loc = 4)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path)
    plt.close()

def save_JSD_estimate(hist, path = 'JSD_estimate.png'):
    x = range(len(hist['D_losses']))
    
    y = hist['JSD_estimate']
    
    plt.plot(x, y, label = 'DCGAN')
    plt.xlabel('Epoch')
    plt.ylabel('JSD estimate')
    
    plt.legend(loc = 4)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path)
    plt.close()

class Discriminator_DC(nn.Module):
    def __init__(self, in_channel = 3, d = 128, mean = 0.0, std = 0.02):
        super(Discriminator_DC, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, d, 4, 2, 1),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(d*8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
        
    def forward(self, x):
        x = self.conv(x)
        # x.size() : [batch_size]
        return x.squeeze()

class Generator_DC(nn.Module):
    def __init__(self, z = 100, d = 128, out_channel = 3, mean = 0.0, std = 0.02):
        super(Generator_DC, self).__init__()
        self.z = z
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(z, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(),
        
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(),
        
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
        
            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        
            nn.ConvTranspose2d(d, out_channel, 4, 2, 1),
            nn.Tanh(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
        
        
    def forward(self, x):
        x = x.view(x.size(0), self.z, 1, 1)
        x = self.deconv(x)
        # x.size() : [batch_size, out_channel, 64, 64]
        return x

    
# model setting
MODE = 'dcgan'

dataset_name = 'celeba'
isCrop = True
data_dir = '../GAN/data/resized_celebA'

D = to_cuda(Discriminator_DC())
G = to_cuda(Generator_DC())

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

D_optimizer = torch.optim.Adam(D.parameters(), lr = lr, betas = (beta1, beta2))
G_optimizer = torch.optim.Adam(G.parameters(), lr = lr, betas = (beta1, beta2))

criterion = nn.BCELoss()

batch_size = 128
image_size = 64
channel = 3

epochs = 30
n_critic = 1
step = 0
generator_step = 0


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['JSD_estimate'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

dir = '%s_%s_results' % (dataset_name, MODE)
if not os.path.isdir(dir):
    os.mkdir(dir)
if not os.path.isdir(dir + '/Fixed_results'):
    os.mkdir(dir + '/Fixed_results')

if dataset_name == 'celeba':
    if isCrop:
        transform = transforms.Compose([
            transforms.Resize(108),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
        
dataset = datasets.ImageFolder(data_dir, transform)
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)


D_labels = to_cuda(torch.ones(batch_size))
D_fakes = to_cuda(torch.ones(batch_size))


# training
print('training started')
training_start_time = time.time()

for epoch in range(epochs):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    
    for x, _ in data_loader:
        if isCrop:
            x = x[:, :, 22:86, 22:86]
            
        step += 1
        
        # Train Discriminator        
        x = to_cuda(x)
        z = to_cuda(torch.randn(batch_size, 100))
                
        d_x = D(x)
        d_z = D(G(z))
        
        D_x_loss = criterion(d_x, D_labels)
        D_z_loss = criterion(d_z, D_fakes)
        
        # Discriminator maximizes log(D(x)) + log(1-D(G(z)))
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        D_losses.append(D_loss)
        
        # Train Generator
        if step % n_critic == 0:
            generator_step += 1
            z = to_cuda(torch.randn(batch_size, 100))
            d_z = D(G(z))
            
            # Generator maximizes log(D(G(z)))
            G_loss = criterion(d_z, D_labels)
            
            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()
    
            G_losses.append(G_loss)
        
        
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
    D_ave_loss = -torch.mean(torch.FloatTensor(D_losses))
    G_ave_loss = torch.mean(torch.FloatTensor(G_losses))
    JSD_estimate = 0.5 * D_ave_loss + math.log10(2) 

    print('[%d/%d] - ptime: %.2f, D_loss: %.3f, G_loss: %.3f' %  
          (epoch + 1, epochs, per_epoch_ptime, D_ave_loss, G_ave_loss))
    
    train_hist['D_losses'].append(D_ave_loss)
    train_hist['G_losses'].append(G_ave_loss)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
    path = dir + '/Fixed_results/%s_%s_' % (dataset_name, MODE) + str(epoch + 1) + '.png'
    save_sample(epoch+1, path = path)

end_time = time.time()
total_ptime = end_time - training_start_time
train_hist['total_ptime'].append(total_ptime)


# Save results
print("Total ptime: %.2f" % total_ptime)

torch.save(D.state_dict(), dir + '/discriminator_param.pkl')
torch.save(G.state_dict(), dir + '/generator_param.pkl')

with open(dir + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
    
save_train_hist(train_hist, path = dir + '/train_hist.png')
save_JSD_estimate(train_hist, path = dir + '/JSD_estimate.png')

# save gif 
images = []
for e in range(epochs):
    img = dir + '/Fixed_results/%s_%s_' % (dataset_name, MODE) + str(e+1) + '.png'
    images.append(imageio.imread(img))
imageio.mimsave(dir + '/generation_animation.gif', images, fps = 5)

