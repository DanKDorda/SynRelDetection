import os
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class ProposalDs(data.Dataset):
    def __init__(self, opts):
        super(ProposalDs, self).__init__()
        self.opts = opts
        # self.data_root = os.path.join(os.getcwd(), 'datasets/all_angles')
        self.data_root = '/Users/i517610/PycharmProjects/SynRelDetection/datasets/all_angles'
        self.paths = os.listdir(self.data_root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, angle):
        im = Image.open(os.path.join(self.data_root, self.paths[angle]))
        im = self.transform(im)
        return im

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    print('testing')
    opts = 'yayo'
    ds = ProposalDs(opts)
    item = ds[0]
    print(type(item), item.shape)
    for i in range(360):
        ds[i]

    print('done')