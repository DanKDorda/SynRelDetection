import sys
import os
from easydict import EasyDict as edict
import yaml
import functools
import tqdm
from pprint import pformat
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import model
from data.visual_genome_loader import VG_dataset, custom_collate


def main(opts):
    dl, dl_val = get_dataloader(opts)

    trainer = model.SyntheticGraphLearner(opts)
    current_iter = 0

    if opts.writer.use_writer:
        writer = SummaryWriter()
        writer.add_text('options', pformat(opts), 0)

    for epoch in range(opts.num_epochs):
        tqdm.tqdm.write(f'Starting epoch: {epoch}')
        # fw
        for data_item in tqdm.tqdm(dl):
            trainer.forward(data_item)
            trainer.compute_loss()
            trainer.optimize_params()
            current_iter += 1

            # logging
            if current_iter % opts.logs.loss_out == 0:
                loss = trainer.get_loss()
                tqdm.tqdm.write(str(round(loss, 2)))
                if opts.writer.use_writer:
                    writer.add_scalar('main supervised loss', loss, global_step=current_iter)
                    if current_iter % opts.logs.im_out == 0:
                        graph_im = trainer.get_image_output()
                        writer.add_image('connectivity_graph', torch.tensor(graph_im), current_iter, dataformats='HWC')

        trainer.scheduler.step()
        if epoch % opts.train.schedule.lr_step_every == 0:
            tqdm.tqdm.write(f'lr set to: {trainer.scheduler.get_lr()}')

        # val loop
        if epoch % opts.logs.val == 0:
            trainer.eval()
            trainer.clear_eval_dict()
            val_iter = 0
            for data_item in tqdm.tqdm(dl_val):
                trainer.evaluate(data_item)
                if val_iter < 5 and opts.writer.use_writer:
                    graph_im = trainer.get_image_output()
                    writer.add_image(f'val_graph/im_{val_iter}', torch.tensor(graph_im), current_iter, dataformats='HWC')
                val_iter += 1
            eval_result = trainer.get_eval_dict()
            sens = eval_result['TP'] / (eval_result['TP'] + eval_result['FN'])
            spec = eval_result['TN'] / (eval_result['TN'] + eval_result['FP'])
            if opts.writer.use_writer:
                writer.add_scalar('val/sensitivity', sens, current_iter)
                writer.add_scalar('val/specificity', spec, current_iter)
            tqdm.tqdm.write(
                f'TP: {eval_result["TP"]}, TN: {eval_result["TN"]}, FP: {eval_result["FP"]}, FN: {eval_result["FN"]}')
            trainer.train()


def get_dataloader(opts):
    ds = VG_dataset(opts, 'train')
    ds_val = VG_dataset(opts, 'val')

    partial_collate_fn = functools.partial(custom_collate, use_shared_memory=opts.num_workers > 0)
    dl = data.DataLoader(ds, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                         collate_fn=partial_collate_fn)

    dl_val = data.DataLoader(ds_val, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                             collate_fn=partial_collate_fn)

    return dl, dl_val


def get_opts(config_path=os.path.join(os.getcwd(), 'options/debug_opts.yaml')):
    with open(config_path) as stream:
        try:
            opts = edict(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    return opts


if __name__ == "__main__":
    # print('test mode')
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print('args: \n', args)
    if args.debug:
        opts = get_opts()
    else:
        cp = os.path.join(os.getcwd(), 'options/easy_bce_2000.yaml')
        opts = get_opts(cp)

    print('options acquired')
    print('================================')

    main(opts)
