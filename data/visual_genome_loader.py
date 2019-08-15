import os
import os.path as osp

from PIL import Image
import json

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms


class VG_dataset(data.Dataset):

    def __init__(self, ds_opts, split='train'):
        self.opts = ds_opts
        self.split = split

        ## bunch of paths
        self.data_root = os.path.join(self.opts.data_root, split)
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

        # define limiter list
        self.limiter_list = json.load(open(osp.join(self.data_root, 'limiter_list.json')))

    def __getitem__(self, idx):
        # randomise the scale

        # get image data
        img_info = self.annotations[idx]
        image_id = str(img_info['id']) + '.jpg'
        image_path = osp.join(self.image_root, image_id)

        # transform image
        img = Image.open(image_path)
        img = self.transform(img)

        # DO THE LIMITING
        num_rels = 10  # len(img_info['relationships'])
        naughty_list = self.limiter_list[idx]
        naughty_list = naughty_list[:int(np.floor(self.opts.limiter.fraction_dropped * num_rels))]
        naughty_adjacency_idcs = torch.zeros(num_rels)
        if self.split == 'train':
            for bad_idx in sorted(naughty_list, key=lambda x: -x):
                naughty_adjacency_idcs[img_info['relationships'].pop(bad_idx)[0]] = 1

        # img_info: dict
        # keys: ['id', 'path', 'height', 'width', 'regions', 'objects', 'relationships']
        return {'path': image_path, 'visual': img, 'image_info': img_info, 'indices_removed': naughty_adjacency_idcs}

    def __len__(self):
        return len(self.annotations)


def custom_collate(batch, use_shared_memory=False):
    out = {}
    out['path'] = [b['path'] for b in batch]

    if use_shared_memory:
        numel = sum([x['visual'].numel() for x in batch])
        storage = batch[0]['visual'].storage()._new_shared(numel)
        out_tensor = batch[0]['visual'].new(storage)
        torch.stack([b['visual'] for b in batch], 0, out=out_tensor)

        numel_2 = sum([x['indices_removed'].numel() for x in batch])
        storage_2 = batch[0]['indices_removed'].storage()._new_shared(numel_2)
        index_tensor = batch[0]['indices_removed'].new(storage_2)
        torch.stack([b['indices_removed'] for b in batch], 0, out=index_tensor)

    out['visual'] = out_tensor
    out['indices_removed'] = index_tensor
    out['objects'] = [b['image_info']['objects'] for b in batch]
    out['relationships'] = [b['image_info']['relationships'] for b in batch]

    return out


class CustomDataLoader(data.DataLoader):
    pass


class CustomIter(data.dataloader._DataLoaderIter):
    pass


if __name__ == '__main__':
    import sys
    from pprint import pprint
    from easydict import EasyDict as edict
    from janky_trainloop import get_dataloader
    import yaml

    print('TEST MODE ENTERED')

    # get a fake config file
    config_pth = '/Users/i517610/PycharmProjects/SynRelDetection/options/debug_opts.yaml'

    with open(config_pth) as stream:
        try:
            opts = edict(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    print('config obtained!')
    pprint(opts)

    # ds = VG_dataset(opts)
    # item1 = VG_dataset[3]

    dl = get_dataloader(opts)

    for i, item in enumerate(dl[0]):
        if i > 3:
            break

        print(item)
