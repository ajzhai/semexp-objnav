import numpy as np
import random
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.ndimage.measurements import center_of_mass
from train import BasicFCN, MapTrajDataset, device
import matplotlib.pyplot as plt

#device = torch.device('cuda:0')

common_cls = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv', 'dining-table', 'oven', 
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'cup', 'bottle']

color_palette = np.array([
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]).reshape((20, 3)) * 255

id_color = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
#         [127, 127, 127],
#         [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


def visualize_obj_preds(pred, cls_ids, z_map, obj_map, mask):
    pred = torch.sigmoid(pred[cls_ids]).cpu().numpy()
    max_p = np.array([np.max(p) for p in pred])
    for i in range(len(cls_ids)):
        pred[i] /= max_p[i]
    
    z_map = z_map.cpu().numpy()
    obj_map = obj_map.cpu().numpy()
    mask = mask.cpu().numpy()
    rgb = np.zeros((pred[0].shape[0], pred[0].shape[1], 3), dtype=float)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            rgb[i, j] = z_map[i, j]  * 0.8
            if np.sum(obj_map[:, i, j]) > 0:
                rgb[i, j] = id_color[np.argmax(obj_map[:, i, j])]/255 
            elif mask[i, j]:
                rgb[i, j] += 0.2
    
    fig, axs = plt.subplots(1, len(cls_ids) + 1, figsize=(4* len(cls_ids), 4))
    axs[0].imshow(np.clip(rgb, 0, 1))
    axs[0].set_title('GT')
    axs[0].axis('off')
    for c in range(len(cls_ids)):
        pred_rgb = np.copy(rgb)
        pred_rgb[np.logical_not(mask.astype(bool))] = 0
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if mask[i, j] < 1:
                    pred_rgb[i, j] += pred[c, i, j] * id_color[cls_ids[c]]/255
                    if np.argmax(obj_map[:, i, j]) == cls_ids[c] and np.sum(obj_map[:, i, j]):
                        pred_rgb[i, j] = [1, 1, 0]
        axs[c + 1].imshow(np.clip(pred_rgb, 0, 1))
        axs[c + 1].axis('off')
        axs[c + 1].set_title('%s (%.4f max)' % (common_cls[cls_ids[c]], max_p[c]))
    
    


with torch.no_grad():
    batch_size = 32
    imsize = 224
    C = 16
    val_dataset = MapTrajDataset('./saved_maps/val/full/', N=40, imsize=imsize,is_train=False, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model = BasicFCN(C + 4, C, imsize=imsize).to(device)
    model.load_state_dict(torch.load('./weights/half_ep40.pth'))
    model.eval()
    for (partial, full, mask) in val_loader:

        obj_preds = model(partial)
        break
    for i in range(32):
        visualize_obj_preds(obj_preds[i], [0, 1, 2, 3, 4, 7, 9], full[i][0], full[i][4:], mask[i][0])
        plt.savefig('plots/qual/%d.png' % i)
        plt.close()
