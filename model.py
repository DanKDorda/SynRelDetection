import numpy as np
import torch
import torch.nn as nn

import copy

import utils


class SyntheticGraphLearner(nn.Module):
    def __init__(self, opts):
        super(SyntheticGraphLearner, self).__init__()
        self.opts = opts
        self.method = opts.train.method
        self.use_cuda = self.opts.train.cuda

        # Declare output properties
        self.adjacency_tensor = torch.FloatTensor()
        self.gt_adjacency_tensor = torch.FloatTensor()
        self.connectivity_matrix = torch.LongTensor()
        self.gt_connectivity_matrix = torch.LongTensor()
        self.predicted_image = torch.FloatTensor()
        self.eval_dict = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        # Define networks
        self.feature_net = FeatureNet(opts)
        self.graph_proposal_net = GraphProposalNetwork(opts)
        self.final_predictor = None

        if self.use_cuda:
            self.feature_net.cuda()
            self.graph_proposal_net.cuda()
            # self.final_predictor.cuda()

        # Define optimizers
        param_list = list(self.feature_net.parameters()) + list(self.graph_proposal_net.parameters())
        self.supervised_optimizer = torch.optim.Adam(param_list, lr=self.opts.train.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.supervised_optimizer,
                                                         self.opts.train.schedule.lr_step_every,
                                                         gamma=self.opts.train.schedule.gamma)

        # Define loss functions
        self.l1_critetion = nn.L1Loss()
        self.softmax_criterion = nn.CrossEntropyLoss()
        self.loss = torch.FloatTensor()

    def forward(self, input_data):
        #### PAIRWISE METHOD ==> LIST OPTION

        # get detections
        # TODO: how does pytorch handle batches of annotations????
        # answer: BADLY
        ct = utils.CompoundTimer()
        image, objects, relationships = input_data['visual'], input_data['objects'], input_data['relationships']
        d_max = max([len(obj) for obj in objects])
        ct.mark('create gt adjacenecy')
        self.gt_adjacency_tensor = utils.rel_list_to_adjacency_tensor(relationships, self.opts.batch_size,
                                                                      d_max)
        self.gt_connectivity_matrix = utils.rel_list_to_connectivity_matrix(relationships, self.opts.batch_size,
                                                                            d_max)
        self.image = image
        self.objects = objects
        ct.mark('create gt adjacenecy end')
        ### MOVE TO GPU
        if self.use_cuda:
            self.gt_adjacency_tensor.cuda()
            image.cuda()

        if self.method == 'unsupervised':
            image_masked, chosen_idx = self.masker(image, annotations)

        # find features from images -> list
        ct.mark('imagelets')
        imagelets_batched = self.get_imagelets(image, objects)
        ct.mark('imagelets end')

        ct.mark('vertex features')
        vertex_feature_list = [self.feature_net(imagelets) for imagelets in imagelets_batched]
        ct.mark('vertex features end')

        # proposal of edges from object features and geometry
        ct.mark('extract geometry')
        geometry_tensor = torch.zeros(self.opts.batch_size, d_max, 4)

        for i, img_obs in enumerate(objects):
            for j, obj in enumerate(img_obs):
                bb = obj['bbox']
                mid_x = int((bb[0] + bb[1]) / 2)
                mid_y = int((bb[2] + bb[3]) / 2)
                size_x = abs(int((bb[0] - bb[1])))
                size_y = abs(int((bb[2] - bb[3])))
                geometry_tensor[i, j, :] = torch.tensor([mid_x, mid_y, size_x, size_y])

        ct.mark('extract geometry end')
        ct.mark('get adjacency')
        self.connectivity_matrix = self.graph_proposal_net(vertex_feature_list, geometry_tensor)
        ct.mark('get adjacency end')

        with torch.no_grad():
            self.connectivity_argmaxed = torch.zeros_like(self.connectivity_matrix)
            self.connectivity_argmaxed.scatter_(2, torch.argmax(self.connectivity_matrix, dim=2, keepdim=True), 1)
        # with torch.no_grad():
        #     re_list = utils.adjacency_tensor_to_rel_list(self.adjacency_tensor)
        # print(ct)
        if self.method == 'unsupervised':
            raise NotImplementedError('haha fool')
            # propose image for missing boy
            predicted_image = self.final_predictor(vertex_feature_list, self.adjacency_tensor, chosen_idx)
            self.predicted_image = predicted_image

    def compute_loss(self):
        if self.method == 'supervised':
            # self.loss = 0 #self.l1_critetion(self.adjacency_tensor, self.gt_adjacency_tensor) * 0.1

            ce_loss = 0
            for row_idx, row in enumerate(self.connectivity_matrix.split(1, dim=1)):
                target = torch.argmax(self.gt_connectivity_matrix[:, row_idx, :], dim=1)
                row.squeeze_(1)
                ce_loss += self.softmax_criterion(row, target)
            self.loss = ce_loss
        elif self.method == 'unsupervised':
            self.loss = self.l1_criterion(self.predicted_image, self.desired_out)

    def optimize_params(self):
        if self.method == 'supervised':
            self.supervised_optimizer.zero_grad()
            self.loss.backward()
            self.supervised_optimizer.step()
        else:
            raise NotImplementedError('only supervised implemented')

    def evaluate(self, input_data):
        self.forward(input_data)

        TP = torch.sum(
            torch.where((self.gt_connectivity_matrix == 1) * (self.connectivity_argmaxed == 1), torch.tensor(1),
                        torch.tensor(0)))
        FP = torch.sum(
            torch.where((self.gt_connectivity_matrix == 0) * (self.connectivity_argmaxed == 1), torch.tensor(1),
                        torch.tensor(0)))
        TN = torch.sum(
            torch.where((self.gt_connectivity_matrix == 0) * (self.connectivity_argmaxed == 0), torch.tensor(1),
                        torch.tensor(0)))
        FN = torch.sum(
            torch.where((self.gt_connectivity_matrix == 1) * (self.connectivity_argmaxed == 0), torch.tensor(1),
                        torch.tensor(0)))

        self.eval_dict['TP'] += TP.item()
        self.eval_dict['FP'] += FP.item()
        self.eval_dict['TN'] += TN.item()
        self.eval_dict['FN'] += FN.item()

    def scheduler_step(self):
        self.scheduler.step()

    def get_eval_dict(self):
        return self.eval_dict

    def clear_eval_dict(self):
        self.eval_dict = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

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

        if self.use_cuda:
            imagelet_base = torch.cuda.FloatTensor
        else:
            imagelet_base = torch.FloatTensor

        for b in range(self.opts.batch_size):
            bboxes = [obj['bbox'] for obj in objects[b]]
            imagelets = imagelet_base(len(bboxes), 3, 96, 96)
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
        # if self.use_cuda:
        #     imagelets.cuda()
        #
        # for i, image_annotation in enumerate(annotations):
        #     for j, detection in enumerate(image_annotation['objects']):
        #         bbox = detection['bbox']
        #         imagelets[i, j, ...] = image[i, :, bbox[2]:bbox[3], bbox[0]:bbox[1]]

        return imagelet_batch

    def get_loss(self):
        return self.loss.detach().item()

    def get_image_output(self):
        sv = utils.SceneVisualiser()
        sv.visualise(self.image[0], self.objects[0], self.connectivity_argmaxed[0])

        return sv.scene


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

        self.resnet = ResnetBlock(128)

        self.conv5 = nn.Conv2d(128, self.opts.FeatNet.feat_out, 4, 1, 0)

        self.av_pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, im_in):
        output = self.relu0(self.bn0(self.conv0(im_in)))
        output = self.relu1(self.bn1(self.conv1(output)))
        output = self.pool1(output)  # 48 x 48
        output = self.relu2(self.bn2(self.conv2(output)))  # 25 x 25
        output = self.relu3(self.bn3(self.conv3(output)))  # 13 x 13
        output = self.pool3(output)
        output = self.relu4(self.bn4(self.conv4(output)))
        # output = self.av_pool(output)
        output = self.resnet(output)
        output = self.conv5(output)
        return output


class GraphProposalNetwork(nn.Module):
    def __init__(self, opts):
        super(GraphProposalNetwork, self).__init__()
        self.opts = opts
        self.N_heads = opts.GPN.N_heads
        self.feat_in = opts.GPN.feat_in
        self.geometry_feat = 4
        self.feat_out = 256
        self.feat_concat = self.feat_out * 2

        self.preliminary_transform = nn.Linear(self.feat_in, self.feat_out)
        self.geometry_add = nn.Sequential(nn.Linear(self.geometry_feat, 32), nn.LeakyReLU(0.02),
                                          nn.Linear(32, self.feat_out))
        self.geometry_mult = nn.Sequential(nn.Linear(self.geometry_feat, 32), nn.LeakyReLU(0.02),
                                           nn.Linear(32, self.feat_out))
        self.attention_net = nn.Sequential(nn.Linear(self.feat_concat, 128), nn.LeakyReLU(0.02),
                                           nn.Linear(128, self.N_heads))
        self.connectivity_net = nn.Sequential(nn.Linear(self.feat_concat, 128), nn.ReLU(),
                                              nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, object_features, scene_geometry, d_max=10):
        # object_features is a list of tensors of D x 128 x 1 x 1
        # input tensor dim = batch x D_max x feature_len

        # visual_features = torch.cat((object_features, scene_geometry), 2)

        # adjacency_tensor = torch.zeros(len(object_features), d_max, d_max, self.N_heads)
        adjacency_tensor = torch.zeros(len(object_features), d_max, d_max)

        embed_vertices = self.preliminary_transform(
            torch.cat([of.unsqueeze(0) for of in object_features]).view(self.opts.batch_size, 10, self.feat_in))
        embed_geometry_m = self.geometry_mult(scene_geometry)
        embed_geometry_a = self.geometry_add(scene_geometry)

        embed_vertices = embed_vertices * embed_geometry_m
        embed_vertices = embed_vertices + embed_geometry_a
        mega_compound_tensor = torch.zeros(len(object_features), d_max, self.feat_concat)

        for batch_idx, batch_vertices in enumerate(embed_vertices.split(1)):
            # batch_vertices.squeeze_(2)
            # batch_vertices.squeeze_(2)
            # geom_cat = torch.cat((batch_vertices, scene_geometry[batch_idx, ...]), dim=1)
            # embed_vertices = self.preliminary_transform(batch_vertices)
            # embed_vertices = embed_vertices * embed_geometry_m[batch_idx]
            # embed_vertices = embed_vertices + embed_geometry_a[batch_idx]
            batch_vertices.squeeze_(0)
            vlist = torch.split(batch_vertices, 1)

            for i, vi in enumerate(vlist):
                compound_tensor = torch.cat((vi, vi), dim=1)
                for j, vj in enumerate([v for k, v in enumerate(vlist) if k != i]):
                    compound_tensor = torch.cat((compound_tensor, torch.cat((vi, vj), dim=1)))
                # compound_tensor = compound_tensor[1:]
                mega_compound_tensor[batch_idx] = compound_tensor
                # edge_vals = self.connectivity_net(compound_tensor)
                # adjacency_tensor[batch_idx, i] = edge_vals.transpose(1, 0)

        big_edge = self.connectivity_net(mega_compound_tensor)
        adjacency_tensor = big_edge.permute(0, 2, 1)
        return adjacency_tensor


class ResnetBlock(nn.Module):
    def __init__(self, num_lay):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Conv2d(num_lay, num_lay, 3, 1, 1)
        self.norm = nn.BatchNorm2d(num_lay)
        self.relu = nn.ReLU()

    def forward(self, x, leakage=0.1):
        return self.relu(self.norm(self.conv(x))) + leakage * x


if __name__ is '__main__':
    print('test mode')

    import json
    import os.path as osp
    import torchvision.transforms as T
    from PIL import Image
    import janky_trainloop


    def peek_at_im(tensor):
        Image.fromarray((np.array(tensor[0] * 255).astype('uint8')).transpose(1, 2, 0)).show()


    # open an image and its annotations
    data_root = '/Users/i517610/PycharmProjects/SynRelDetection/datasets/sr3/train'
    annotations = json.load(open(osp.join(data_root, 'scene_info.json')))
    lucky_idx = np.random.choice(len(annotations))

    im = Image.open(osp.join(data_root, 'images', str(lucky_idx) + '.jpg'))
    transform = T.Compose([T.ToTensor()])
    im_torch = transform(im)
    im_torch = im_torch.unsqueeze(0)
    lucky_annotation = annotations[lucky_idx]
    data = {'visual': im_torch,
            'objects': [lucky_annotation['objects']],
            'relationships': [lucky_annotation['relationships']]}

    opts = janky_trainloop.get_opts()
    opts.batch_size = 1
    model = SyntheticGraphLearner(opts)
    model.eval()
    model.evaluate(data)

    img = model.get_image_output()
    Image.fromarray(img).show()
