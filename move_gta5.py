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

to_dir = os.path.join("data", "trainA")
if not os.path.exists(to_dir):
    os.mkdir(to_dir)


class GTA5(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(321, 321), crop_size=(321, 321), flip=False, random_crop=False, ignore_label=255, gen_edge=False, edge_radius=3):
        super(GTA5, self).__init__()
        self.root = root
        self.list_path = list_path
        self.resize_size = resize_size
        self.ignore_label = ignore_label
        self.edge_radius = edge_radius
        self.flip = flip
        self.gen_edge = gen_edge

        self.img_ids = [i_id.strip() for i_id in open(os.path.join(root, list_path))]
        for name in self.img_ids:
            img_file = "images/%s" % name
            label_file = "labels/%s" % name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        image = Image.open(os.path.join(self.root, 'gta5', datafiles["img"])).convert('RGB')
        image.save(os.path.dir(to_dir, name))
        return 0

