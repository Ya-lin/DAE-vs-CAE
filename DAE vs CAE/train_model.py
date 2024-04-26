from tqdm import tqdm
import torch
from torch.autograd.functional import jacobian
def AE_trainer(args, ae, optimizer, 
               train_loader, valid_loader):
    device = args.device
    history = {"train loss":[], "valid loss":[]}
    for e in tqdm(range(args.epoch), colour="red"):
        ae.train()
        train_loss = 0
        for x,_ in train_loader:
            x = x.to(device)
            if args.noise_factor is None:
                x_hat = ae(x)
            else:
                noise = torch.randn(*x.shape).to(args.device)
                noisy_x=x+args.noise_factor*noise
                noisy_x = torch.clamp(noisy_x, 0., 1.)
                x_hat = ae(noisy_x)
            optimizer.zero_grad()
            loss = args.loss(x_hat, x)
            if args.cae is not None:
                jab = jacobian(ae.encoder, x, create_graph=True)
                jab = torch.flatten(jab, start_dim=1)
                f_norm = torch.sum(jab**2)/args.batch
                loss += args.cae*f_norm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        history["train loss"].append(train_loss/len(train_loader))
    
        valid_loss = 0
        ae = ae.eval()
        with torch.no_grad():
            for x,_ in valid_loader:
                x = x.to(device)
                if args.noise_factor is None:
                    x_hat = ae(x)
                else:
                    noise = torch.randn(*x.shape).to(args.device)
                    noisy_x=x+args.noise_factor*noise
                    noisy_x = torch.clamp(noisy_x, 0., 1.)
                    x_hat= ae(noisy_x)
                loss = args.loss(x_hat, x)
                valid_loss += loss.item()
        history["valid loss"].append(valid_loss/len(valid_loader))

    return history



