#%% import packages
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
from loguru import logger

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from build_model import AE
from train_model import AE_trainer
from test_model import img2img_hat
from data_process import get_mnist, split_data


#%% set GPU
torch.set_num_threads(8)
if torch.cuda.is_available():
    device = "cuda"
    memory=torch.cuda.get_device_properties(device).total_memory/1024**3
    logger.info(f"gpu memory: {memory}")
else:
    device = "cpu"
    logger.info("gpu is not availabel.")


#%% set hyperparameters
def Args():
    parser = argparse.ArgumentParser()
    args_add = parser.add_argument
    args_add("--device", default=device)
    args_add("--ratio", default=0.8,
             help="train:valid=0.8:0.2")
    args_add("--noise_factor", default=0.5,
             help="noise variance")
    args_add("--c_hid", default=8,
             help="output channel in the first layer")
    args_add("--dim", default=32,
             help="latent dimension in AE")
    args_add("--act", default=nn.GELU(), 
             help="activation function in AE")
    args_add("--loss", default=nn.MSELoss(), 
             help="loss function in AE")
    args_add("--lr", default=1e-4,
             help="learning rate in AE")
    args_add("--batch", default=512,
             help="batch size in AE")
    args_add("--cae", default=None,
             help="whether contractive")
    args_add("--epoch", default=200,
             help="train epoch in AE") 
    args = parser.parse_args()
    return args

args = Args()
logger.info(f"hyperparameters: {args}")

if args.cae is not None: args.batch = 16


#%% Train AE
train, test = get_mnist()
train, valid = split_data(train, args.ratio)

ae = AE(args.c_hid, args.dim, args.act).to(args.device)
optimizer = optim.AdamW(ae.parameters(), lr=args.lr)
train_loader = DataLoader(train, batch_size=args.batch,
                          shuffle=True, drop_last=True)
valid_loader = DataLoader(valid, batch_size=args.batch,
                          shuffle=False, drop_last=True)
history = AE_trainer(args, ae, optimizer, 
                     train_loader, valid_loader)


#%% show training and validation loss
pfig = Path.cwd().joinpath("fig")
pfig.mkdir(exist_ok=True)
ae.eval()
fig = plt.figure()
plt.plot(history["train loss"], label="train")
plt.plot(history["valid loss"], label="valid")
plt.legend()
plt.show()
fig.savefig(pfig.joinpath("train_valid_loss.png"))
plt.close()
logger.info(f'AE loss: {history["train loss"][-1]},\
            {history["valid loss"][-1]}')


#%% check out the result
test_loader = DataLoader(test, batch_size=args.batch)
data_iter = iter(test_loader)
images, labels = next(data_iter)
noise = torch.randn(*images.shape)
noisy_imgs = images + args.noise_factor*noise
noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
with torch.no_grad():
    output = ae(noisy_imgs.to(args.device))
    output = output.cpu().numpy()

noisy_imgs = noisy_imgs.numpy()

fig = img2img_hat(noisy_imgs, output)
fig.savefig(pfig.joinpath("results.png"))
plt.close()



