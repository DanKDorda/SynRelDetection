import os
import os.path as osp

from PIL import Image
import json

import torch
import torch.utils.data as data
from torchvision import transforms


class VG_dataset(data.Dataset):

    def __init__(self, ds_opts):
        self.opts = ds_opts

        ## bunch of paths
        self.data_root = self.opts.data_root
        self.image_root = osp.join(self.data_root, 'images')
        annotation_pth = osp.join(self.data_root, self.opts.path)

        ## load annotations
        self.annotations = json.load(open(annotation_pth))

        # transforms
        # self.image_scales = self.opts.scales

        normalise = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalise
        ])

    def __getitem__(self, idx):
        # randomise the scale
        # TODO:randomise scales

        # get image data
        img_info = self.annotations[idx]
        image_id = str(img_info['id']) + '.jpg'
        image_path = osp.join(self.image_root, image_id)

        # transform image
        img = Image.open(image_path)
        img = self.transform(img)

        # img_info: dict
        # keys: ['id', 'path', 'height', 'width', 'regions', 'objects', 'relationships']
        return {'path': image_path, 'visual': img, 'image_info': img_info}

    def __len__(self):
        return len(self.annotations)


class CustomDataLoader(data.DataLoader):
    pass


class CustomIter(data.dataloader._DataLoaderIter):
    pass
