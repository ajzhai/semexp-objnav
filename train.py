import numpy as np
import random
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

device = torch.device('cuda:8')


class MapTrajDataset(Dataset):
    def __init__(self, data_path, device, N=200, is_train=True, imsize=480, padding=0.25):
        self.is_train = is_train
        self.data_path = data_path
        self.partials = 4
        
        self.imsize = imsize
        self.size_tf = transforms.Resize((imsize, imsize))
        self.aug = transforms.Compose([
            transforms.RandomCrop(imsize, padding=int(padding * imsize)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(180)
        ])
        self.all_maps = []
        N = min(N, len(os.listdir(data_path)))
        for i, fname in enumerate(os.listdir(data_path)[:N]):
            print('load', i)
            fname = 'f%05d.npy' % i
            self.all_maps.append(np.load(os.path.join(self.data_path, fname))[:, :, 48:-48, 48:-48])
            #self.all_maps = [np.load(os.path.join(self.data_path, fname)) for fname in os.listdir(data_path)]
        
        self.N = len(self.all_maps) * self.partials
        self.n_obj_cls = 14
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        fname = 'f%05d.npy' % (idx // self.partials)
        #map_file = np.load(os.path.join(self.data_path, fname))
        map_file = self.all_maps[idx // self.partials]
        maps = torch.tensor(map_file[[idx % self.partials, -2]], device=device)
        maps[0][1] = (maps[0][1] > 0.2).float()
        maps = self.size_tf(maps)
        if self.is_train:
            maps = self.aug(maps)
            
        
        return maps[0], maps[1], maps[0][1].unsqueeze(0)  # mask


class BasicFCN(nn.Module):
    def __init__(self, in_channels, out_channels, imsize=128):
        super(BasicFCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        
        resnet = models.resnet18(pretrained=False)
        self.features16 = nn.Sequential(*list(resnet.children())[1:7])
        self.features16.requires_grad = True
        
        self.features32 = list(resnet.children())[7]
        self.features32.requires_grad = True
        
        self.head16 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0),
            #nn.Sigmoid()
        )
        
        self.head32 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0),
        )
        
        self.double_ups = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final_ups = nn.Upsample(size=(imsize, imsize), mode='bilinear')
       
    def forward(self, x):
        x = self.conv1(x)
        x16 = self.features16(x)
        x32 = self.features32(x16)
        x = self.final_ups(self.head16(x16) + self.head32(self.double_ups(x32)))
        return x
    
    def forward32(self, x):
        x = self.conv1(x)
        x = self.features16(x)
        x = self.features32(x)
        x = self.final_ups(self.head32(x))
        return x
    
    
if __name__ == '__main__':
    batch_size = 32
    imsize = 128
    train_dataset = MapTrajDataset('./saved_maps/full/', imsize=imsize, device=device)
    val_dataset = MapTrajDataset('./saved_maps/val/full/', is_train=False, imsize=imsize, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print(len(train_dataset), 'train,', len(val_dataset), 'val')
    
    C = 16
    model = BasicFCN(C + 4, C, imsize=imsize).to(device)
    
    bce_crit = nn.BCEWithLogitsLoss(pos_weight=10.* torch.ones(C, imsize, imsize, dtype=torch.float).to(device))
    criterion = bce_crit
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 100
    start = time.time()
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_room_loss, train_obj_loss = 0, 0
        for i, (partial, full, mask) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            obj_preds = model(partial)
            
            loss = bce_crit(obj_preds, full[:, 4:] * (1 - mask)) 
            loss.backward()
            optimizer.step()

            train_loss += float(loss) / len(train_loader)
            #print(i)
            
        val_loss = 0.0
        val_room_loss, val_obj_loss = 0, 0
        model.eval()
        with torch.no_grad():
            for (partial, full, mask) in val_loader:

                obj_preds = model(partial)
            
                loss = bce_crit(obj_preds, full[:, 4:] * (1 - mask)) 

                val_loss += float(loss) / len(val_loader)
               
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if epoch % 10 == 9:
            print('Epoch: {}, train loss (obj|room|total): {:.4f} | {:.4f} | {:.4f}, val loss: {:.4f} | {:.4f} | {:.4f} {:.3f}'.format(
                epoch + 1, train_obj_loss, train_room_loss, train_loss, val_obj_loss, val_room_loss, val_loss, time.time() - start))

            torch.save(model.state_dict(), 'weights/half_ep' + str(epoch + 1) + '.pth')
    np.save('tl_half.npy', np.array(train_hist))
    np.save('vl_half.npy', np.array(val_hist))
