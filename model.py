import numpy as np
import torch
import torch.nn as nn

import copy


class SyntheticGraphLearner(nn.Module):
    def __init__(self, opts):
        super(SyntheticGraphLearner, self).__init__()
        self.opts = opts
        self.feature_net = FeatureNet()
        self.edge_proposal_net = GraphProposalNetwork()
        self.final_predictor = ...

    def forward(self, input_data):
        # get detections
        # TODO: how does pytorch handle batches of annotations????
        image, annotations = input_data['visuals'], input_data['image_info']
        image_masked, chosen_idx = self.masker(image, annotations)

        # find features from images -> list
        imagelets = self.get_imagelets(image, annotations)

        # chosen_one = {'object_id': ..., 'imagelet': ..., 'orientation': ..., 'col': ..., 'relationships': ...}
        # object_features = self.feature_net(image_masked, boxes_masked)
        # TODO: see if imagelets can be concatanated into a tensor with an extra dim
        vertex_features = [self.feature_net(imagelet) for imagelet in imagelets]

        # proposal of edges from object features and geometry
        geometry_tensor = torch.zeros(self.opts.batch_size, self.D_max, 4)

        for i, img in enumerate(annotations):
            for j, obj in enumerate(img['objects']):
                bb = obj['bbox']
                mid_x = int((bb[0] + bb[1])/2)
                mid_y = int((bb[2] + bb[3])/2)
                size_x = abs(int((bb[0] - bb[1])))
                size_y = abs(int((bb[2] - bb[3])))
                geometry_tensor[i, j, :] = torch.tensor([mid_x, mid_y, size_x, size_y])

        edge_set = self.edge_proposal_net(vertex_features, geometry_tensor)

        # propose image for missing boy
        predicted_image = self.final_predictor(vertex_features, edge_set, chosen_one)

        self.predicted_image = predicted_image

    def compute_loss(self):
        self.loss = self.l2_criterion(self.predicted_image, self.desired_out)

    @staticmethod
    def masker(image, annotations):
        # TESTING REQUIRED
        # NOT SURE THIS WORKS AS INTENDED
        # ALSO POTENTIAL PROBLEM OF

        image_masked = copy.deepcopy(image)
        masked_objects = []
        # pretend it's a list of dicts
        for i, image_annotation in enumerate(annotations):
            masked_object_idx = np.random.choice(len(image_annotation['objects']))
            bb = annotations['objects'][masked_object_idx]['bbox']
            image_masked[i, :, bb[2]:bb[3], bb[0]:bb[1]] = 0
            masked_objects.append(masked_object_idx)

        return image_masked, masked_objects

    def get_imagelets(self, image, annotations):
        # list option
        bboxes = [obj['bbox'] for obj in annotations['objects']]
        imagelets = [image[..., bb[2]:bb[3], bb[0]:bb[1]] for bb in bboxes]

        # tensor_option
        # feat dimension: Batch x Dmax X d
        # imagelet dim: Batch x Dmax X c x h x w
        D_max = max([len(a['objects']) for a in annotations])
        c = 3
        h = 32
        w = 32
        imagelets = torch.zeros(self.opts.batch_size,D_max, c, w, h)

        if self.opts.cuda:
            imagelets.cuda()

        for i, image_annotation in enumerate(annotations):
            for j, detection in enumerate(image_annotation['objects']):
                bbox = detection['bbox']
                imagelets[i, j, ...] = image[i, :, bbox[2]:bbox[3], bbox[0]:bbox[1]]

        return imagelets


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.relu3 = nn.ReLU()

    def forward(self, input):
        output = self.relu0(self.conv0(input))
        output = self.relu1(self.conv1(output))
        output = self.relu2(self.conv2(output))
        output = self.relu3(self.conv3(output))
        return output


class GraphProposalNetwork(nn.Module):
    def __init__(self):
        super(GraphProposalNetwork, self).__init__()
        self.D_max = ...
        self.N_heads = ...
        self.N_edges = ...

        visual_attention_layers = []

        self.get_visual_attention = nn.Sequential(*visual_attention_layers)

        # embed feed forward, [conv, batch, relu] x N -> softmax |||| OR |||| use MLP?
        embed_attention_layers = []

        self.embed_attention = nn.Sequential(*embed_attention_layers)

    def forward(self, object_features, scene_geometry):
        # oh no, these bad boys are different sizes across the batch.... >:9
        # object_features is a list -> turn it into a tensor
        # input tensor dim = batch x D_max x feature_len

        visual_features = torch.cat((object_features, scene_geometry), 2)

        incidence_column = torch.zeros(self.opts.batch_size, self.N_heads, self.D_max, 1)
        incidence_matrix = torch.zeros(self.opts.batch_size, self.N_heads, self.D_max, self.N_edges)

        for i in range(self.N_edges):
            h = self.get_visual_attention(visual_features, incidence_column)
            b = visual_features*h.T
            incidence_column = self.embed_attention(b)
            incidence_matrix = torch.cat([incidence_matrix, incidence_column], dim=3)

        # generate columns of incidence matrix, based on previous columns, or previous visual_matrix_attentions
        # do that N times

        # build up multiple incidence matrices
        # multiply them out into multiple adjacency matrices
        # concat them together together into an adjacency tensor

        adjacency_tensor = torch.matmul(incidence_matrix, incidence_matrix.transpose(2, 3))
        return adjacency_tensor


if __name__ is '__main__':
    print('test mode')

    import json
    import os.path as osp
    import torchvision.transforms as T
    from PIL import Image

    def peek_at_im(tensor):
        Image.fromarray((np.array(tensor[0] * 255).astype('uint8')).transpose(1, 2, 0)).show()

    # open an image and its annotations
    data_root = '/Users/i517610/PycharmProjects/SynRelDetection/datasets/synthrel1'
    annotations = json.load(open(osp.join(data_root, 'scene_info.json')))
    lucky_idx = np.random.choice(len(annotations))

    im = Image.open(osp.join(data_root, str(lucky_idx) + '.jpg'))
    transform = T.Compose([T.ToTensor()])
    im_torch = transform(im)
    im_torch = im_torch.unsqueeze(0)
    lucky_annotation = annotations[lucky_idx]

    model = SyntheticGraphLearner()
    model.forward([im_torch, lucky_annotation])
