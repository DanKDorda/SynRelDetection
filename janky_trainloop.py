import sys
import os
from easydict import EasyDict as edict
import yaml

import torch
import torch.nn as nn
import torch.utils.data as data

import model
from data.visual_genome_loader import VG_dataset


def main(opts):

    dl = get_dataloader(opts)

    for data in dl:
        print(data)

    trainer = model.SyntheticGraphLearner(opts)

    for epoch in range(opts.num_epochs):
        for data in dl:
            trainer.forward(data)
            trainer.calculate_loss()
            trainer.optimize_params()


def get_dataloader(opts):
    ds = VG_dataset(opts)
    dl = data.DataLoader(ds, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    return dl


if __name__ == "__main__":
    print('test mode')

    config_path = '/Users/i517610/PycharmProjects/SynRelDetection/options/debug_opts.yaml'
    with open(config_path) as stream:
        try:
            opts = edict(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    print('options acquired')

    main(opts)
