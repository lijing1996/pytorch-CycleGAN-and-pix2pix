import os
import torch
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from tqdm import tqdm
to_dir = os.path.join("data", "trainB")
if not os.path.exists(to_dir):
    os.mkdir(to_dir)


class Cityscapes(data.Dataset):
    def __init__(self, root, list_path):
        super(Cityscapes, self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_list = [line.strip().split() for line in open(os.path.join(root, list_path))]

        self.files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            self.files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(os.path.join(self.root, 'cityscapes', datafiles["img"])).convert('RGB')
        image.save(os.path.join(to_dir, name) + ".png")
        return 0


dataset = Cityscapes("data", "list/cityscapes/train.lst")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
for _ in tqdm(dataloader):
    pass