import numpy as np
import torch
import torch.nn as nn

import copy

import utils


class SyntheticGraphLearner(nn.Module):
    def __init__(self, opts):
        super(SyntheticGraphLearner, self).__init__()
        self.opts = opts
        self.method = opts.method

        # Declare output properties
        self.adjacency_tensor = torch.FloatTensor()
        self.gt_adjacency_tensor = torch.FloatTensor()
        self.predicted_image = torch.FloatTensor()

        # Define networks
        self.feature_net = FeatureNet(opts)
        self.graph_proposal_net = GraphProposalNetwork(opts)
        self.final_predictor = None

        # Define optimizers
        if self.opts.cuda:
            self.feature_net.cuda()
            self.graph_proposal_net.cuda()
            self.final_predictor.cuda()

        param_list = list(self.feature_net.parameters()) + list(self.graph_proposal_net.parameters())
        self.supervised_optimizer = torch.optim.Adam(param_list, lr=opts.lr)

        # Define loss functions
        self.l1_critetion = nn.L1Loss()
        self.softmax_criterion = nn.CrossEntropyLoss()
        self.loss = torch.FloatTensor()

    def forward(self, input_data):
        #### PAIRWISE METHOD ==> LIST OPTION

        # get detections
        # TODO: how does pytorch handle batches of annotations????
        # answer: BADLY
        image, objects, relationships = input_data['visual'], input_data['objects'], input_data['relationships']
        d_max = max([len(obj) for obj in objects])
        self.gt_adjacency_tensor = utils.rel_list_to_adjacency_tensor(relationships, self.opts.batch_size,
                                                                      d_max)

        if self.method == 'unsupervised':
            image_masked, chosen_idx = self.masker(image, annotations)

        # find features from images -> list
        imagelets_batched = self.get_imagelets(image, objects)

        vertex_feature_list = [self.feature_net(imagelets) for imagelets in imagelets_batched]

        # proposal of edges from object features and geometry
        geometry_tensor = torch.zeros(self.opts.batch_size, d_max, 4)

        for i, img_obs in enumerate(objects):
            for j, obj in enumerate(img_obs):
                bb = obj['bbox']
                mid_x = int((bb[0] + bb[1]) / 2)
                mid_y = int((bb[2] + bb[3]) / 2)
                size_x = abs(int((bb[0] - bb[1])))
                size_y = abs(int((bb[2] - bb[3])))
                geometry_tensor[i, j, :] = torch.tensor([mid_x, mid_y, size_x, size_y])

        self.adjacency_tensor = self.graph_proposal_net(vertex_feature_list, geometry_tensor)

        if self.method == 'unsupervised':
            # propose image for missing boy
            predicted_image = self.final_predictor(vertex_feature_list, self.adjacency_tensor, chosen_idx)
            self.predicted_image = predicted_image

    def compute_loss(self):
        if self.method == 'supervised':
            self.loss = self.l1_critetion(self.adjacency_tensor, self.gt_adjacency_tensor)
        elif self.method == 'unsupervised':
            self.loss = self.l1_criterion(self.predicted_image, self.desired_out)

    def optimize_params(self):
        if self.method == 'supervised':
            self.supervised_optimizer.zero_grad()
            self.loss.backward()
            self.supervised_optimizer.step()
        else:
            raise NotImplementedError('only supervised implemented')

    @staticmethod
    def masker(image_tensor, object_batch):
        # TESTING REQUIRED
        # NOT SURE THIS WORKS AS INTENDED
        # ALSO POTENTIAL PROBLEM OF

        image_masked = copy.deepcopy(image_tensor)
        masked_objects = []
        # pretend it's a list of dicts
        for i, object_list in enumerate(object_batch):
            masked_object_idx = np.random.choice(len(object_list))
            bb = object_list[masked_object_idx]['bbox']
            image_masked[i, :, bb[2]:bb[3], bb[0]:bb[1]] = 0
            masked_objects.append(masked_object_idx)

        return image_masked, masked_objects

    def get_imagelets(self, image, objects):
        # list option
        imagelet_batch = []

        for b in range(self.opts.batch_size):
            bboxes = [obj['bbox'] for obj in objects[b]]
            imagelets = torch.zeros(len(bboxes), 3, 96, 96)
            for i, bb in enumerate(bboxes):
                imagelets[i, ...] = image[b, :, bb[2]:bb[3], bb[0]:bb[1]]
            imagelet_batch.append(imagelets)

        # # tensor_option
        # # feat dimension: Batch x Dmax X d
        # # imagelet dim: Batch x Dmax X c x h x w
        # D_max = max([len(a['objects']) for a in annotations])
        # c = 3
        # h = 32
        # w = 32
        # imagelets = torch.zeros(self.opts.batch_size,D_max, c, w, h)
        #
        # if self.opts.cuda:
        #     imagelets.cuda()
        #
        # for i, image_annotation in enumerate(annotations):
        #     for j, detection in enumerate(image_annotation['objects']):
        #         bbox = detection['bbox']
        #         imagelets[i, j, ...] = image[i, :, bbox[2]:bbox[3], bbox[0]:bbox[1]]

        return imagelet_batch

    def get_loss(self):
        return self.loss.detach().item()


class FeatureNet(nn.Module):
    def __init__(self, opts):
        super(FeatureNet, self).__init__()
        self.opts = opts

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=8)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU()

        self.av_pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, input):
        output = self.relu0(self.bn0(self.conv0(input)))
        output = self.relu1(self.bn1(self.conv1(output)))
        output = self.pool1(output)  # 48 x 48
        output = self.relu2(self.bn2(self.conv2(output)))  # 25 x 25
        output = self.relu3(self.bn3(self.conv3(output)))  # 13 x 13
        output = self.pool3(output)
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.av_pool(output)
        return output


class GraphProposalNetwork(nn.Module):
    def __init__(self, opts):
        super(GraphProposalNetwork, self).__init__()
        self.opts = opts
        self.N_heads = opts.GPN.N_heads
        self.feat_in = opts.GPN.feat_in
        self.feat_out = 128
        self.feat_concat = self.feat_out * 2

        self.preliminary_transform = nn.Sequential(nn.Linear(self.feat_in, self.feat_out))
        self.attention_nets = torch.nn.ModuleList()
        for i in range(self.N_heads):
            attention_layers = [nn.Linear(self.feat_concat, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1)]
            attention_net = nn.Sequential(*attention_layers)
            self.attention_nets.append(attention_net)

    def forward(self, object_features, scene_geometry, d_max=10):
        # oh no, these bad boys are different sizes across the batch.... >:9
        # object_features is a list -> turn it into a tensor
        # input tensor dim = batch x D_max x feature_len

        # visual_features = torch.cat((object_features, scene_geometry), 2)

        adjacency_tensor = torch.zeros(len(object_features), self.N_heads, d_max, d_max)

        for batch_idx, batch_vertices in enumerate(object_features):
            batch_vertices.squeeze_(2)
            batch_vertices.squeeze_(2)
            embed_vertices = self.preliminary_transform(batch_vertices)
            vlist = torch.split(embed_vertices, 1)

            for i, vi in enumerate(vlist):
                for j, vj in enumerate([v for k, v in enumerate(vlist) if k != i]):
                    for h in range(self.N_heads):
                        compound_tensor = torch.cat((vi, vj), dim=1)
                        edge_val = self.attention_nets[h](compound_tensor)
                        adjacency_tensor[batch_idx, h, j, i] = edge_val

        # for i, vi in enumerate(feature_list):
        #
        #     for j, vj in enumerate(feature_list[:i, i + 1:]):
        #         vi_embed = self.preliminary_transform(torch.cat([vi, scene_geometry[]]))
        #         vj_embed = self.preliminary_transform(vj)
        #
        #         for h in range(self.N_heads):
        #             adjacency_tensor[h, j, i] = self.attention_net[h](torch.cat((vi_embed, vj_embed)))

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
