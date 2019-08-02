import torch
import time
from pprint import pformat


def bbox_to_slice(bbox):
    bbox_slice = ...
    return bbox_slice


def rel_list_to_adjacency_tensor(rels, batch_size, num_objects):
    A = torch.zeros(batch_size, 2, num_objects, num_objects)
    for i, rel in enumerate(rels):
        for r in rel:
            A[i, 0, r[0], r[2]] = r[1][0]
            A[i, 1, r[0], r[2]] = r[1][1]

    return A


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
