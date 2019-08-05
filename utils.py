import torch
import time

def rel_list_to_adjacency_tensor(relation_list_batch, batch_size, num_objects):
    A = torch.zeros(batch_size, 2, num_objects, num_objects)
    for i, rel in enumerate(relation_list_batch):
        for r in rel:
            A[i, 0, r[0], r[2]] = r[1][0]
            A[i, 1, r[0], r[2]] = r[1][1]

    return A


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
