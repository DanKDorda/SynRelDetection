import torch
import time
import numpy as np
from PIL import Image
import cv2


def rel_list_to_adjacency_tensor(relation_list_batch, batch_size, num_objects):
    A = torch.zeros(batch_size, num_objects, num_objects, 2)
    for i, rel in enumerate(relation_list_batch):
        for r in rel:
            A[i, r[0], r[2], 0] = r[1][0]
            A[i, r[0], r[2], 1] = r[1][1]

    return A


def rel_list_to_connectivity_matrix(relation_list_batch, batch_size, num_objects):
    A = torch.zeros(batch_size, num_objects, num_objects)
    for i, rel in enumerate(relation_list_batch):
        for r in rel:
            A[i, r[0], r[2]] = 1

    return A


# TODO: fix for new adjacency tensor spec
def adjacency_tensor_to_rel_list(at):
    rl = []
    at_batched = torch.split(at, 1)
    for atb in at_batched:
        rlb = []
        atb = atb.squeeze(0)
        for row_idx, row in enumerate(torch.split(atb, 1, dim=1)):
            elem1 = row_idx
            row = row.squeeze(1)
            for col_idx, col in enumerate(row.split(1, dim=1)):
                col = col.squeeze(1)
                elem2 = col_idx
                relations = col
                rlb.append((elem1, relations, elem2))
        rl.append(rlb)
    return rl


class SceneVisualiser:
    def __init__(self):
        self.scene = np.zeros((500, 500, 3))

    def visualise(self, nodes_or_image, geometry, adjacency):
        centres = []
        for meta in geometry:
            bb = meta['bbox']
            centre = ((bb[0]+bb[1])/2, (bb[2]+bb[3])/2)
            centres.append(centre)

        if isinstance(nodes_or_image, list):
            raise ValueError('ya rong')
        elif isinstance(nodes_or_image, torch.Tensor):
            self.scene = (nodes_or_image.permute(1, 2, 0)*255).numpy().astype('uint8').copy()

        line_ends = []
        for i, row in enumerate(adjacency.split(1, dim=0)):
            for j, entry in enumerate(row.split(1, dim=1)):
                if entry == 1:
                    pt1 = tuple([int(n) for n in centres[i]])
                    pt2 = tuple([int(n) for n in centres[j]])
                    line_ends.append((pt1, pt2))

        self.add_line(line_ends)

    def add_nodes(self, nodes, geometry):
        for node, meta in zip(nodes, geometry):
            bb = meta['bbox']
            self.scene[:, bb[2]:bb[3], bb[0]:bb[1]] = node

    def add_line(self, line_ends):
        for pt_pair in line_ends:
            cv2.line(self.scene, pt_pair[0], pt_pair[1], color=(255, 0, 0), thickness=3)

    def show_im(self):
        scene_np = self.scene.astype('uint8')
        im_pil = Image.fromarray(scene_np)
        im_pil.show()


class CompoundTimer:

    def __init__(self):
        self.time_array = []
        self.labels = []
        # self.mark('init')

    def mark(self, label=None):
        t = time.time()
        self.time_array.append(t)

        if label is None:
            label = len(self.time_array)
        self.labels.append(label)

    def __str__(self):
        time_diff = [t1 - t0 for t0, t1 in zip(self.time_array[:-1], self.time_array[1:])]
        time_diff.insert(0, 0)
        t_total = self.time_array[-1] - self.time_array[0]
        strings = [f'    {d:.2f}s\n{l} - {d/t_total*100:.1f}%' for d, l in zip(time_diff, self.labels)]
        strings = chr(10).join(strings)
        return strings


if __name__ == '__main__':
    print('test yo')
    import janky_trainloop

    opts = janky_trainloop.get_opts('/Users/i517610/PycharmProjects/SynRelDetection/options/debug_opts.yaml')
    dl = janky_trainloop.get_dataloader(opts)

    item = next(iter(dl))
    img, obj, rels = item['visual'], item['objects'], item['relationships']

    connec = rel_list_to_connectivity_matrix(rels, opts.batch_size, 10)

    sv = SceneVisualiser()
    sv.visualise(img[0], obj[0], connec[0])
    sv.show_im()
