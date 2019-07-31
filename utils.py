import torch


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
