import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, c_hid, dim, act,
                 c_in = 1,
                 width = 28,
                 height = 28,
                ):
        super().__init__()
        self.example_input = torch.zeros(2, c_in, width, height)
        self.act = act
        # layers in encoder
        self.en_cv1 = nn.Conv2d(c_in, c_hid, 3, stride=2, padding=1)
        self.en_cv2 = nn.Conv2d(c_hid, 2*c_hid, 3, stride=2, padding=1)
        self.en_bn2 = nn.BatchNorm2d(2*c_hid)
        self.en_cv3 = nn.Conv2d(2*c_hid, 4*c_hid, 3, stride=2, padding=0)
        self.en_flat = nn.Flatten(start_dim=1)
        self.en_fc1 = nn.Linear(4*c_hid*(width//2//2//2)*(height//2//2//2),
                                16*c_hid)
        self.en_fc2 = nn.Linear(16*c_hid, 4*c_hid)
        self.en_fc3 = nn.Linear(4*c_hid, dim)
        # layers in decoder
        self.de_fc1 = nn.Linear(dim, 4*c_hid)
        self.de_fc2 = nn.Linear(4*c_hid, 16*c_hid)
        self.de_fc3 = nn.Linear(16*c_hid, 
                                4*c_hid*(width//2//2//2)*(height//2//2//2))
        self.de_unflat = nn.Unflatten(dim=1,unflattened_size=
                                      (4*c_hid,width//2//2//2,height//2//2//2))
        self.de_cv1 = nn.ConvTranspose2d(4*c_hid, 2*c_hid, 3, 
                                         stride=2, output_padding=0)
        self.de_bn1 = nn.BatchNorm2d(2*c_hid)
        self.de_cv2 = nn.ConvTranspose2d(2*c_hid, c_hid, 3, 
                                         stride=2,padding=1,output_padding=1)
        self.de_bn2 = nn.BatchNorm2d(c_hid)
        self.de_cv3 = nn.ConvTranspose2d(c_hid, c_in, 3, 
                                         stride=2,padding=1,output_padding=1)     
        
    def encoder(self, x):
        x = self.act(self.en_cv1(x))
        x = self.act(self.en_bn2(self.en_cv2(x)))
        x = self.act(self.en_cv3(x))
        x = self.en_flat(x)
        x = self.act(self.en_fc1(x))
        x = self.act(self.en_fc2(x))
        x = self.en_fc3(x)
        return x

    def decoder(self, x):
        x = self.act(self.de_fc1(x))
        x = self.act(self.de_fc2(x))
        x = self.act(self.de_fc3(x))
        x = self.de_unflat(x)
        x = self.act(self.de_bn1(self.de_cv1(x)))
        x = self.act(self.de_bn2(self.de_cv2(x)))
        x = self.de_cv3(x)
        x = torch.sigmoid(x)
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



