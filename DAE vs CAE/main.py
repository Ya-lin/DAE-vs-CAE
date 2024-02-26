#%% Import packages
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from MNIST_AE import AE
from Trainer import AE_trainer
from Data import MNIST_loader, Split_data

#%% GPU
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
gpu_memory = torch.cuda.get_device_properties(device).total_memory/1024**3

#%% Set Hyperparameters
def Args():
    parser = argparse.ArgumentParser()
    args_add = parser.add_argument
    args_add("--device", default=device)
    args_add("--int", default=None,
             help="labels of normal samples")
    args_add("--ratio", default=0.8,
             help="train:valid=0.8:0.2")
    args_add("--noise_factor", default=0.5,
             help="noise variance")
    args_add("--basech", default=8,
             help="number of output channel in the first layer")
    args_add("--dim", default=32,
             help="latent dimension in AE")
    args_add("--act", default=nn.GELU(),
             help="activation function in AE")
    args_add("--loss",default=nn.MSELoss(),
             help="loss function in AE")
    args_add("--lr",default=1e-4,
             help="learning rate in AE")
    args_add("--batch", default=512,
             help="batch size in AE")
    args_add("--cae", default=None,
             help = "whether contractive")
    args_add("--epoch", default=200, 
             help="train epoch in AE") 
    args = parser.parse_args()
    return args

args = Args()
print("\nHyperparameters: \n", args)

#%% Train AE
train, test = MNIST_loader()
train, valid = Split_data(train, args.ratio)

ae = AE(args).to(args.device)
optimizer = optim.AdamW(ae.parameters(), lr=args.lr)
train_loader = DataLoader(train, batch_size=args.batch,
                          shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch,
                          shuffle=False, drop_last=True)
history = AE_trainer(args, ae, optimizer, 
                     train_loader, valid_loader)


#%% Show training and validation loss
ae.eval()
   
plt.plot(history["train loss"],label="train")
plt.plot(history["valid loss"],label="valid")
plt.legend()
plt.show()
print("AE loss: ", history["train loss"][-1], 
      history["valid loss"][-1])


#%% Check out the results
test_loader = DataLoader(test, batch_size=args.batch)
data_iter = iter(test_loader)
images, labels = next(data_iter)
noise = torch.randn(*images.shape)
noisy_imgs = images + args.noise_factor*noise
noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

output = ae(noisy_imgs.to(args.device))
output = output.detach().cpu().numpy()

noisy_imgs = noisy_imgs.numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, 
                         sharex=True, sharey=True, 
                         figsize=(25,4))
for noisy_imgs, row in zip([noisy_imgs, output], axes):
    for img, ax in zip(noisy_imgs, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)




