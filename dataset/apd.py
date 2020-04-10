#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import os
import torch.utils.data as data

import torch
import torchvision.transforms as transforms

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = cv2.resize(img, (112,112), interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except:
        print('Cannot load image ' + path)
        return np.zeros((112,112,3), dtype=np.uint8)

class APD(data.Dataset):
    def __init__(self, root, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader
        self.img_basenames = [ x for x in os.listdir(root) if len(x.split('_'))>=2 and x.lower().endswith(('.jpg','.png','.bmp'))]

    def __getitem__(self, index):

        base_name = self.img_basenames[index]
        full_path = self.root + '/' + base_name
        img = self.loader(full_path)
        imglist = [img, cv2.flip(img, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])
            imglist.append(os.path.splitext(base_name)[0])
            return imglist
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            imgs.append(os.path.splitext(base_name)[0])
            return imgs

    def __len__(self):
        return len(self.img_basenames)


if __name__ == '__main__':
    root = '/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/APD/C'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = APD(root, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(len(data))
        for d in data[:2]:
            print(len(d), d[0].shape)
        print(len(data[2]), data[2][0])
