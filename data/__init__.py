if __name__ == '__main__':
    # messy way to handle imports
    import sys

    import sys
    import yaml
    from easydict import EasyDict as edict
    from pprint import pprint

import torch
import torch.utils.data as data
from data.visual_genome_loader import VG_dataset


def get_dataloaders(opts, split='train_val', shuffle=True):
    train_loader = None
    val_loader = None

    if 'train' in split:
        train_dset = VG_dataset(opts.train)
        train_loader = data.DataLoader(train_dset, batch_size=opts.train.batch_size, shuffle=shuffle,
                                       num_workers=opts.train.num_workers)

    if 'val' in split:
        val_dset = VG_dataset(opts.val)
        val_loader = data.DataLoader(val_dset, batch_size=opts.val.batch_size, shuffle=shuffle,
                                     num_workers=opts.val.num_workers)

    return {'train': train_loader, 'val': val_loader}


if __name__ == '__main__':
    print('TEST MODE ENTERED')

    # get a fake config file
    config_pth = '/Users/i517610/PycharmProjects/testing_ground/options/debug_opts.yaml'

    with open(config_pth) as stream:
        try:
            opts = edict(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    print('config obtained!')
    pprint(opts)

    dls = get_dataloaders(opts.data, split='train')

    for idx, data in enumerate(dls['train']):
        print('======================')
        print(idx)
        if idx == 1:
            print(data.keys())

        if idx > 10:
            print('====== almost complete ======')
            break

    print('test completed')
