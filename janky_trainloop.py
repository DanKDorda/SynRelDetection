import sys
import os
from easydict import EasyDict as edict
import yaml
import functools
import tqdm
from pprint import pformat

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import model
from data.visual_genome_loader import VG_dataset, custom_collate


def main(opts):
    dl = get_dataloader(opts)

    trainer = model.SyntheticGraphLearner(opts)
    current_iter = 0

    if opts.writer.use_writer:
        writer = SummaryWriter()
        writer.add_text('options', pformat(opts), 0)

    for epoch in range(opts.num_epochs):
        for data in tqdm.tqdm(dl):
            trainer.forward(data)
            trainer.compute_loss()
            trainer.optimize_params()
            current_iter += 1

            if current_iter % opts.logs.loss_out == 0:
                loss = trainer.get_loss()
                if opts.writer.use_writer:
                    writer.add_scalar('supervised l1 loss', loss, global_step=current_iter)
                tqdm.tqdm.write(str(loss))


def get_dataloader(opts):
    ds = VG_dataset(opts)

    partial_collate_fn = functools.partial(custom_collate, use_shared_memory=opts.num_workers > 0)
    dl = data.DataLoader(ds, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                         collate_fn=partial_collate_fn)
    return dl


if __name__ == "__main__":
    print('test mode')

    config_path = os.path.join(os.getcwd(), 'options/debug_opts.yaml')
    with open(config_path) as stream:
        try:
            opts = edict(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    print('options acquired')

    main(opts)
